use csveee::{ParserBuilder, RecordTerminator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = ParserBuilder::new()
        .delimiter(b',')
        .quote(Some(b'"'))
        .terminator(RecordTerminator::LF)
        .has_headers(true)
        .concurrency(4)
        .chunk_size(256 * 1024)
        .build();

    let sum = parser.parse(
        "suites/fixtures/standard.csv",
        || 0u64,
        |sum, [_name, age, _city]| {
            *sum += age.parse::<u64>()?;
            Ok(())
        },
        |sums| sums.iter().sum::<u64>(),
    )?;

    println!("Total age: {sum}");
    Ok(())
}
