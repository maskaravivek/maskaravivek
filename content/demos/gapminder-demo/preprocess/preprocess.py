import pandas as pd

df = pd.read_csv('../data/countries_regions.csv')


def preprocess_file(file_name):
    data = pd.read_csv('../data/raw/' + file_name)

    forest_years = data.columns

    processed = []
    for index, row in data.iterrows():
        country = row['country']
        selectedCountry = df.loc[df['name'] == country]

        if len(selectedCountry) == 0:
            continue

        region = selectedCountry.iloc[0]['World bank region']
        geo = selectedCountry.iloc[0]['geo']
        for year in forest_years[1:]:
            item = {
                "country": country,
                "region": region,
                "geo": geo,
                "year": year,
                "value": row[year]
            }
            processed.append(item)

    processed = pd.DataFrame(processed)
    processed.to_csv('../data/processed/' + file_name)


preprocess_file('child_mortality_0_5_year_olds_dying_per_1000_born.csv')
preprocess_file('children_and_elderly_per_100_adults.csv')
preprocess_file('children_per_woman_total_fertility.csv')
preprocess_file('life_expectancy_years.csv')
preprocess_file('population_density_per_square_km.csv')
