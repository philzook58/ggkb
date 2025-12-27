// https://github.com/philippeitis/rustnomial

// instead of big, use checked versions that panic?
use std::collections::HashMap;

use ggkb::{cleanup, saturate, simple_parse};

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    file: PathBuf,
}
fn main() {
    let cli = Cli::parse();
    println!("File path: {:?}", cli.file);
    let file = std::fs::read_to_string(cli.file).expect("Failed to read file");
    println!("File contents:\n{}", file);

    let (names, eqs) = simple_parse(&file).expect("Failed to parse input");
    /* .or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        })
        .unwrap();
    */
    let revnames = names
        .iter()
        .map(|(k, v)| (v.clone(), k.clone()))
        .collect::<HashMap<ggkb::Id, String>>();
    let rules = saturate(eqs);
    let rules = cleanup(&rules);
    println!("Saturated rules:");
    for (lhs, rhs) in rules {
        println!(
            "{:?} -> {:?}",
            lhs.iter()
                .map(|i| revnames.get(i).unwrap().clone())
                .collect::<Vec<_>>(),
            rhs.iter()
                .map(|i| revnames.get(i).unwrap().clone())
                .collect::<Vec<_>>()
        );
    }
}
