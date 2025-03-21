import pandas as pd

season_stats = pd.DataFrame(
    [
        {
            "cluster_name": "very high",
            "freq_rel": 0.2065,
            "mean_tourists": 4585.8421052631575,
            "std_tourists": 600.7764663941985,
            "mean_excursionists": 6001.631578947368,
            "std_excursionists": 910.7123713839693,
        },
        {
            "cluster_name": "high",
            "freq_rel": 0.2609,
            "mean_tourists": 4019.5416666666665,
            "std_tourists": 425.7224018134318,
            "mean_excursionists": 3653.0416666666665,
            "std_excursionists": 790.854874784884,
        },
        {
            "cluster_name": "mid",
            "freq_rel": 0.2935,
            "mean_tourists": 2915.1111111111113,
            "std_tourists": 426.69213665965236,
            "mean_excursionists": 2476.6296296296296,
            "std_excursionists": 829.2709099957879,
        },
        {
            "cluster_name": "low",
            "freq_rel": 0.2391,
            "mean_tourists": 1798.090909090909,
            "std_tourists": 480.97192365250527,
            "mean_excursionists": 1165.909090909091,
            "std_excursionists": 480.7615891766995,
        },
    ]
).set_index("cluster_name")

weather_stats = pd.DataFrame(
    [
        [0.15, 140.30952381, 463.13601314, -773.23809524, 1187.05786413],
        [0.20, 128.32142857, 319.20470369, 31.10714286, 824.51775728],
        [0.65, 72.86263736, 406.09163233, 282.91208791, 839.91419686],
    ],
    columns=["freq_rel", "mean_tourists", "std_tourists", "mean_excursionists", "std_excursionists"],
    index=["bad", "unsettled", "good"],
)

weekday_stats = pd.DataFrame(
    [
        [-362.65934066, 112.47000414, -391.15384615, 791.84158997],
        [-265.34065934, 122.13648742, -352.27472527, 937.65181712],
        [-147.92307692, 247.5534754, -465.62637363, 733.9652083],
        [92.23076923, 233.81638923, -380.35164835, 779.58869239],
        [678.05494505, 149.44551186, -232.91208791, 500.50732948],
        [320.46938776, 204.41996226, 590.98979592, 797.28902804],
        [-278.93406593, 193.01797188, 1240.27472527, 1149.59709071],
    ],
    columns=["mean_tourists", "std_tourists", "mean_excursionists", "std_excursionists"],
    index=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
)

season = {s: season_stats.loc[s, "freq_rel"] for s in season_stats.index}
weather = {w: weather_stats.loc[w, "freq_rel"] for w in weather_stats.index}
weekday = weekday_stats.index


def tourist_presences_stats(weekday, season, weather):
    # Season
    mean = season_stats.loc[season, "mean_tourists"]
    std2 = season_stats.loc[season, "std_tourists"] ** 2
    # Weather
    mean += weather_stats.loc[weather, "mean_tourists"]
    std2 += weather_stats.loc[weather, "std_tourists"] ** 2
    # Weekdays
    mean += weekday_stats.loc[weekday, "mean_tourists"]
    std2 += weekday_stats.loc[weekday, "std_tourists"] ** 2
    # Finalize and return
    return {"mean": mean, "std": std2 ** (1 / 2)}


def excursionist_presences_stats(weekday, season, weather):
    # Season
    mean = season_stats.loc[season, "mean_excursionists"]
    std2 = season_stats.loc[season, "std_excursionists"] ** 2
    # Weather
    mean += weather_stats.loc[weather, "mean_excursionists"]
    std2 += weather_stats.loc[weather, "std_excursionists"] ** 2
    # Weekdays
    mean += weekday_stats.loc[weekday, "mean_excursionists"]
    std2 += weekday_stats.loc[weekday, "std_excursionists"] ** 2
    # Finalize and return
    return {"mean": mean, "std": std2 ** (1 / 2)}
