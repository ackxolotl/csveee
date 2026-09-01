//! The two output modes: `bytes()` hands the callback `&mut [u8]` and
//! skips UTF-8 validation, `flexible()` hands it a slice instead of a
//! fixed-size array so records may vary in length.
//!
//! The declared arity is important information for the parser: a chunk
//! parsed from the wrong state producing records of the wrong width is
//! rejected on the spot. `flexible()` gives that check up, so the
//! accumulator is the only oracle left — see below.

use csveee::ParserBuilder;

// records vary in length: a sensor id followed by its readings
const READINGS: &str = "\
sensor,readings
lab-1,20.4,20.9,21.1
lab-2,18.0
roof,25.5,25.9,26.4,27.0
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // the parser skips UTF-8 validation, handing out &mut [u8]
    let mut bytes = ParserBuilder::new().bytes().build();

    let max_age = bytes.parse(
        "suites/fixtures/standard.csv",
        || 0u32,
        |max, [_name, age, _city]| {
            let age: u32 = std::str::from_utf8(age)?.parse()?;
            *max = (*max).max(age);
            Ok(())
        },
        |maxes| maxes.iter().copied().max().unwrap_or(0),
    )?;

    println!("Max age: {max_age}");

    // the parser accepts records of any length – passing a zero-information
    // accumulation function (e.g., counting records with no type conversions
    // or field count checking) to the parser can lead to sequential and thus
    // expensive reparsing at merge time
    let mut flexible = ParserBuilder::new().flexible().build();

    let peak = flexible.parse_slice(
        READINGS,
        || f64::MIN,
        |peak, fields| {
            if fields.len() < 2 || fields[0].is_empty() {
                return Err("expected a sensor id and at least one reading".into());
            }
            for reading in &mut fields[1..] {
                *peak = peak.max(reading.parse()?);
            }
            Ok(())
        },
        |peaks| peaks.iter().copied().fold(f64::MIN, f64::max),
    )?;

    println!("Peak reading: {peak}");
    Ok(())
}
