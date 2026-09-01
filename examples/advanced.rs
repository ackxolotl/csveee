use csveee::ParserBuilder;

struct Columns {
    names: Vec<String>,
    ages: Vec<u32>,
    cities: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = ParserBuilder::new().concurrency(4).build();

    // collect into columnar vectors, then merge across chunks
    let result = parser.parse(
        "suites/fixtures/standard.csv",
        || Columns {
            names: vec![],
            ages: vec![],
            cities: vec![],
        },
        |state, [name, age, city]| {
            state.names.push(name.to_owned());
            state.ages.push(age.parse()?);
            state.cities.push(city.to_owned());
            Ok(())
        },
        |states| {
            let mut merged = Columns {
                names: vec![],
                ages: vec![],
                cities: vec![],
            };
            for state in states {
                merged.names.append(&mut state.names);
                merged.ages.append(&mut state.ages);
                merged.cities.append(&mut state.cities);
            }
            merged
        },
    )?;

    println!("Names: {:?}", result.names);
    println!("Ages: {:?}", result.ages);
    println!("Cities: {:?}", result.cities);

    Ok(())
}
