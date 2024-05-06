use std::collections::{HashMap, HashSet};

use fancy_regex::Regex;
use regex::Regex as Regex2;

pub struct Tokenizer {
    dataset: String,
    regex_pattern:  String,
    special_tokens: HashMap<String, usize>,
    inverse_special_tokens: HashMap<usize, String>,
    vocab: HashMap<usize, Vec<u8>>,
    merges: HashMap<(u8, u8), usize>,
    desired_vocab_size: usize,
    verbose: bool, 
}

impl Tokenizer {
    pub fn create(
        dataset: String,
        desired_vocab_size: usize,
        split_pattern: String,
        special_tokens: Vec<String>,
        verbose: bool,
    ) -> Tokenizer {
        let mut t = Tokenizer {
            dataset,
            inverse_special_tokens: HashMap::new(),
            merges: HashMap::new(),
            desired_vocab_size: desired_vocab_size,
            regex_pattern: split_pattern,
            special_tokens: HashMap::new(),
            verbose,
            vocab: HashMap::new(),
        };

        t.train();
        
        let special_token_start = 1038257;
        let mut compiled_tokens: HashMap<String, usize> = HashMap::new();
        for (i, token) in special_tokens.iter().enumerate() {
            compiled_tokens.insert(token.to_string(), special_token_start + i);
        }

        t.register_special_tokens(compiled_tokens);

        return t
    }

    pub fn train(&mut self) {
        if self.verbose {
            println!("Training tokenizer...");
        }

        let text = &self.dataset;
        
        let dataset_vocab_size = text.chars().collect::<HashSet<_>>().len();
        
        if self.verbose {
            println!("Dataset vocab size: {:?}", dataset_vocab_size);
            println!("Desired vocab size: {:?}", dataset_vocab_size);
        }

        if self.desired_vocab_size < dataset_vocab_size {
            panic!("Dataset vocab size size is larger than desired vocab size. Please increase the desired vocab size.");
        }

        let num_merges = self.desired_vocab_size - dataset_vocab_size;

        // Split the text into text chunks based on the regex pattern
        let text_chunks: Vec<&str> = Regex::new(&self.regex_pattern)
            .unwrap()
            .find_iter(text)
            .map(|m| m.unwrap().as_str())
            .collect();

        // Input text preprocessing
        let mut tokens: Vec<Vec<u8>> = text_chunks.iter()
            .map(|chunk| chunk.as_bytes().to_vec())
            .collect();

        // Iteratively merge the most common pairs to create new tokens
        let mut merges = HashMap::new(); // (usize, usize) -> usize
        let mut vocab: HashMap<usize, Vec<u8>> = HashMap::new();   // usize -> Vec<u8>
        
        // Initialize the vocabulary with the individual bytes
        for (i, &byte) in text.as_bytes().iter().enumerate() {
            vocab.insert(i, vec![byte]);
        }

        // Merge the most common pairs
        for i in 0..num_merges {
            // Count the number of times every consecutive pair appears
            let mut stats = HashMap::new();
            for chunk_tokens in &tokens {
                stats = self.get_stats(chunk_tokens, &mut stats);
            }
            
            // Find the pair with the highest count
            let pair = *stats.iter().max_by_key(|(_, &count)| count).unwrap().0;
            
            // Mint a new token: assign it the next available id
            let token = dataset_vocab_size + i;
            
            // Replace all occurrences of pair in tokens with token
            tokens = tokens.iter()
                .map(|chunk_tokens| self.merge(chunk_tokens, pair, token))
                .collect();

            // Save the merge
            merges.insert(pair, token);
            
            // Save the vocabulary
            let mut merged_bytes = Vec::new();

            merged_bytes.extend_from_slice(&vocab[&(pair.0 as usize)]);
            merged_bytes.extend_from_slice(&vocab[&(pair.1 as usize)]);
            vocab.insert(token, merged_bytes);

            
            // Print merge details if verbose mode is enabled
            if self.verbose {
                println!("merge {}/{}: {:?} -> {} ({:?}) had {} occurrences",
                    i + 1, num_merges, pair, token, vocab[&token], stats[&pair]);
            }
        }

        // Save class variables
        self.merges = merges; // Used in encode()
        self.vocab = vocab;   // Used in decode()
    }


    pub fn register_special_tokens(
        &mut self,
        special_tokens: HashMap<String, usize>,
    ) {
        self.special_tokens = special_tokens.clone();
        self.inverse_special_tokens = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        // Decode the user's desired handling of special tokens
        let special: HashMap<String, usize> = self.special_tokens.clone();

        if special.is_empty() {
            // Shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text);
        }

        // Handle special tokens by splitting the text based on their occurrence
        let special_pattern = format!("({})", special.keys().map(|s| regex::escape(s)).collect::<Vec<_>>().join("|"));
        let special_chunks: Vec<&str> = regex::Regex::new(&special_pattern)
            .unwrap()
            .split(text)
            .collect();

        // Encode chunks of text separately
        let mut tokens = Vec::new();
        for part in special_chunks {
            if let Some(&token_id) = special.get(part) {
                // This is a special token, encode it separately
                tokens.push(token_id);
            } else {
                // This is an ordinary sequence, encode it normally
                tokens.extend(self.encode_ordinary(part));
            }
        }
        tokens
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut part_bytes = Vec::new();
        for &token in tokens {
            if let Some(token) = self.vocab.get(&token) {
                part_bytes.extend(token.iter().copied());
            } else if let Some(token) = self.inverse_special_tokens.get(&token) {
                part_bytes.extend(token.bytes());
            } else {
                // panic!(format!("Invalid token id: {}", token));
            }
        }
        String::from_utf8_lossy(&part_bytes).into_owned()
    }

    fn get_stats(&self, tokens: &[u8], counts: &mut HashMap<(u8, u8), usize>) -> HashMap<(u8, u8), usize> {
        for pair in tokens.iter().zip(tokens.iter().skip(1)) {
            *counts.entry((*pair.0, *pair.1)).or_insert(0) += 1;
        }
        counts.to_owned().clone()
    }

    fn merge(&self, tokens: &[u8], pair: (u8, u8), token: usize) -> Vec<u8> {
        let mut new_tokens = Vec::with_capacity(tokens.len());
        let mut i = 0;

        while i < tokens.len() {
            // Check if the current position is not at the very last position
            // and if the pair matches
            if tokens[i] == pair.0 && i < tokens.len() - 1 && tokens[i + 1] == pair.1 {
                new_tokens.push(token as u8);
                i += 2;
            } else {
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }

        new_tokens
    }


    fn encode_chunk(&self, text_bytes: &[u8]) -> Vec<usize> {
        let mut tokens: Vec<usize> = text_bytes.iter().map(|&byte| byte as usize).collect();

        while tokens.len() >= 2 {
            // Find the pair with the lowest merge index
            let mut stats = HashMap::new();
            stats = self.get_stats(&tokens.iter().map(|&x| x as u8).collect::<Vec<_>>(), &mut stats);
            let pair = *stats.iter().min_by_key(|(&p, &count)| (self.merges.get(&p).unwrap_or(&usize::MAX), count)).unwrap().0;

            if !self.merges.contains_key(&pair) {
                break; // Nothing else can be merged anymore
            }

            // Merge the best pair (lowest merge index)
            let token = self.merges[&pair];
            tokens = self.merge(&tokens.iter().map(|&x| x as u8).collect::<Vec<_>>(), pair, token as usize).iter().map(|&x| x as usize).collect();
        }

        tokens
    }


    fn encode_ordinary(&self, text: &str) -> Vec<usize> {
        // Split text into chunks of text by categories defined in regex pattern
        let text_chunks: Vec<&str> = Regex::new(&self.regex_pattern)
            .unwrap()
            .find_iter(text)
            .map(|m| m.unwrap().as_str())
            .collect();
        
        // Encode each chunk separately
        let mut tokens = Vec::new();
        for chunk in text_chunks {
            let chunk_bytes = chunk.as_bytes();
            let chunk_tokens = self.encode_chunk(chunk_bytes);
            tokens.extend(chunk_tokens);
        }

        tokens
    }
}