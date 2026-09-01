//! The two entry points for input that is not a file on disk:
//! `parse_slice` for bytes already in memory and `parse_stream` for
//! readers without random access. See `simple.rs` for `parse`.

use std::io::Cursor;

use csveee::Parser;

const DATA: &str = "\
name,age,city
Ada,36,London
Grace,45,New York
Alan,41,Cambridge
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = Parser::new();

    // in-memory input: chunked and parsed in parallel, no copy
    let oldest = parser.parse_slice(
        DATA,
        || None::<(String, u32)>,
        |oldest, [name, age, _city]| {
            let age: u32 = age.parse()?;
            if oldest.as_ref().is_none_or(|&(_, best)| age > best) {
                *oldest = Some((name.to_string(), age));
            }
            Ok(())
        },
        |states| {
            states
                .iter_mut()
                .filter_map(|s| s.take())
                .max_by_key(|&(_, age)| age)
        },
    )?;

    match oldest {
        Some((name, age)) => println!("Oldest: {name} ({age})"),
        None => println!("No records."),
    }

    // any `Read`, parsed sequentially: stdin, a socket, ...
    let (count, sum) = parser.parse_stream(
        Cursor::new(DATA),
        || (0u64, 0u64),
        |(count, sum), [_name, age, _city]| {
            *count += 1;
            *sum += age.parse::<u64>()?;
            Ok(())
        },
        |states| {
            states
                .iter()
                .copied()
                .fold((0, 0), |(c, s), (c2, s2)| (c + c2, s + s2))
        },
    )?;

    if count == 0 {
        println!("No records.");
    } else {
        println!("rows: {count}, mean age: {:.1}", sum as f64 / count as f64);
    }
    Ok(())
}
