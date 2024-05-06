use std::fs;

// use fancy_regex::Regex;

use tch::{nn::{self, LinearConfig, Module}, Device, Tensor};

use crate::tokenizer::Tokenizer;
mod tokenizer;
mod splitter;
mod types;


fn main() {
    let read = fs::read_to_string("./assets/input.txt").unwrap();
    // Change according to your device
    let device = Device::cuda_if_available();
    // let device = Device::Cpu;

    // Tokenizer
    // // --- Parameters --------------
    let vocab_size = 300;
    let gpt4_split_pattern = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
    
    // // --- Training --------------
    let tokeniser = Tokenizer::create(
        read.clone(),
        vocab_size,
        gpt4_split_pattern.to_string(),
        vec![],
        true,
    );
    // --------------------------

    // Encoding and splitting
    let tokens = tokeniser.encode(&read);

    let dataset = splitter::split_dataset(
        tokens, 
        (0.8, 0.1, 0.1),
        device,
    );
    // --------------------------


    // Model
    // // Model Hyperparameters --------------
    let batch_size = 8;
    let learning_rate = 0.01;
    let context_length = 32;
    let max_iterations = 1000;
    let eval_interval = 100;
    let eval_iterations = 10;
    let n_embedding = 128;
    let n_head = 2;
    let n_hidden = 256;
    let n_layer = 2;
    let dropout = 0.1;

    let seed = 42;
    
    let vs = nn::VarStore::new(tch::Device::cuda_if_available());
    
    // // Batch Generator --------------------------
    fn batch_generator(mode: String) -> (Tensor, Tensor) {
        let data: Tensor  = match mode.as_str() {
            "eval" => dataset.x_eval,
            "test" => dataset.x_test,
            _ => panic!("Invalid mode"),
        };

        let ix = Tensor::randint(len(data) - context_length, &[batch_size], (Kind::Int64, device));
        let x_tensor = ix.iter().map(| i | data.narrow(0, i, i+context_length)).unwrap();
        let y_tensor = ix.iter().map(| i | data.narrow(0, i+1, i+context_length+1)).unwrap();
        let x = Tensor::stack(&vec![tensors],0);
        let y = Tensor::stack(&vec![tensors],0);
        x = x.to(device);
        y = y.to(device);

        return (x, y);
    }
    // --------------------------------------------

    // // Self Attention --------------------------
    struct CausalSelfAttention {
        c_attention: nn::Linear,
        c_projection: nn::Linear,
        attention_dropout: f64,
        residual_dropout: f64,
        bias: Tensor,
        n_head: i64,
        n_embedding: i64,
        head_size: i64,
    }
    
    impl Compute for CausalSelfAttention {
        fn new<'a, P>(vs: &'a nn::Path, n_embedding: i64, n_head: i64, head_size: i64, context_length: i64, dropout: f64) -> CausalSelfAttention {
            assert!(n_embedding % n_head == 0);
            
            // key, query, value projections for all heads, but in a batch
            let c_attention = nn::linear(vs, n_embedding, 3 * n_embedding, LinearConfig { bias: false, ..Default::default() }).build(vs);
            
            // output projection
            let c_projection = nn::linear(vs, n_embedding, n_embedding, LinearConfig { bias: false, ..Default::default() }).build(vs);
            
            // causal mask to ensure that attention is only applied to the left in the input sequence
            let bias = Tensor::tril(&Tensor::ones(&[1, 1, context_length, context_length], (tch::Kind::Float, tch::Device::cuda_if_available())), 0).unwrap();
            
            CausalSelfAttention {
                c_attention,
                c_projection,
                attention_dropout: dropout,
                residual_dropout: dropout,
                bias,
                n_head,
                n_embedding,
                head_size,
            }
        }
    
        fn forward(&self, x: &Tensor) -> Tensor {
            let (b, t, c) = x.size3().unwrap();
    
            // calculate query, key, values for all heads in batch and move head forward to be the batch dim
            let (q, k, v) = self.c_attention.forward(x).split_with_sizes(&[self.n_embedding; 3], 2);
            let k = k.view([b, t, self.n_head, self.head_size]).transpose(1, 2); // (B, nh, T, hs)
            let q = q.view([b, t, self.n_head, self.head_size]).transpose(1, 2); // (B, nh, T, hs)
            let v = v.view([b, t, self.n_head, self.head_size]).transpose(1, 2); // (B, nh, T, hs)
    
            // causal self-attention
            let scores = q.matmul(&k.transpose(-2, -1)) * (1.0 / (self.head_size as f64).sqrt());
            let mut att = scores.masked_fill(&self.bias.slice(2, 0, t, 1).eq(0), std::f64::NEG_INFINITY);
            let softmax = att.softmax(-1);
            let attention_dropout = softmax.dropout(self.attention_dropout, self.training());
            let value_output = attention_dropout.matmul(&v); // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            let y = value_output.transpose(1, 2).reshape(&[b, t, self.n_embedding]); // re-assemble all head outputs side by side
    
            // output projection
            let out = self.residual_dropout(self.c_projection.forward(&y), self.training());
            out
        }
    }
    // --------------------------------------------

    // // Feed Forward ----------------------------
    struct FeedForward {
        net: nn::Sequential,
    }
    
    impl FeedForward {
        fn new<'a, P>(vs: &'a nn::Path, n_embedding: i64, n_hidden: i64, dropout: f64) -> FeedForward {
            let net = nn::sequential!(
                vs,
                nn::Linear::new(n_embedding, n_hidden, Default::default()),
                nn::Func::relu(),
                nn::Linear::new(n_hidden, n_embedding, Default::default()),
                nn::Dropout::new(dropout),
            );
            
            FeedForward { net }
        }
    
        fn forward(&self, x: &Tensor) -> Tensor {
            self.net.forward(x)
        }
    }    
    // --------------------------------------------

    // // Transformer Block -----------------------
    struct Block {
        self_attention: CausalSelfAttention,
        feed_foward: FeedForward,
        attention_normalization: nn::LayerNorm,
        feed_foward_normalization: nn::LayerNorm,
    }
    
    impl Compute for Block {
        fn new<'a, P>(vs: &'a nn::Path, n_embedding: i64, n_head: i64, n_hidden: i64, dropout: f64, context_length: i64) -> Block {
            let self_attention = CausalSelfAttention::new(vs, n_embedding, n_head, n_embedding / n_head, context_length, dropout);
            let feed_foward = FeedForward::new(vs, n_embedding, n_hidden, dropout);
            let attention_normalization = nn::layer_norm(vs, vec![n_embedding], Default::default());
            let feed_foward_normalization = nn::layer_norm(vs, vec![n_embedding], Default::default());
    
            Block { self_attention, feed_foward, attention_normalization, feed_foward_normalization }
        }
    
        fn forward(&self, x: &Tensor) -> Tensor {
            let residual = x.copy();
            let x = x + &self.self_attention.forward(&self.attention_normalization.forward(&x));
            let x = x + &self.feed_foward.forward(&self.feed_foward_normalization.forward(&x));
            x + &residual
        }
    }
    
    // --------------------------------------------

    // // Optimizer ------------------------------
    fn adam_optimizer(&mut self, learning_rate: f32) {
        let mut g = Tensor::new();
        const BETA:f32 = 0.9;

        let mut velocity = Tensor::zeros(&[self.size as i64], (Kind::Float, Device::Cpu)).split(1, 0);
        let mut mom = Tensor::zeros(&[self.size as i64], (Kind::Float, Device::Cpu)).split(1, 0);
        let mut vel_corr = Tensor::zeros(&[self.size as i64], (Kind::Float, Device::Cpu)).split(1, 0);
        let mut mom_corr = Tensor::zeros(&[self.size as i64], (Kind::Float, Device::Cpu)).split(1, 0);
        let mut counter = 0;

        self.values
        .iter_mut()
        .for_each(|t| {
            if t.requires_grad() {
                g = t.grad();
                mom[counter] = BETA * &mom[counter] + (1.0 - BETA) * &g;
                velocity[counter] = BETA * &velocity[counter] + (1.0 - BETA) * (&g.pow(&Tensor::from(2)));    
                mom_corr[counter] = &mom[counter]  / (Tensor::from(1.0 - BETA).pow(&Tensor::from(2)));
                vel_corr[counter] = &velocity[counter] / (Tensor::from(1.0 - BETA).pow(&Tensor::from(2)));

                t.set_data(&(t.data() - learning_rate * (&mom_corr[counter] / (&velocity[counter].sqrt() + 0.0000001))));
                t.zero_grad();
            }
            counter += 1;
        });

    }
    // // --------------------------------------------

    // // GPT Model ------------------------------
    struct GPT {
        tok_emb: nn::Embedding,
        pos_emb: nn::Embedding,
        drop: f64,
        blocks: Vec<TransformerBlock>,
        head: nn::Linear,
    }
    // // --------------------------------------------

    // // Loss Function ---------------------------
    struct LossFunction {
        criterion: nn::CrossEntropyLoss,
    }
    // // --------------------------------------------

    // // Training Loop ---------------------------
    fn train() {

    }
    // // --------------------------------------------

    println!("{:?}", dataset.x_train);
    println!("{:?}", dataset.x_train.size());
}
