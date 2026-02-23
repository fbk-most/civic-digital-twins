"""Example of Bologna mobility scenario.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy import stats

from civic_digital_twins.dt_model import Index, TimeseriesIndex, UniformDistIndex
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.internal.sympyke import Piecewise

vehicle_inflow = np.array(
    [
        221.25,
        208.70090169,
        196.59507377,
        184.93251626,
        173.71322915,
        162.93721244,
        152.60446612,
        142.71499021,
        133.2687847,
        124.26584959,
        115.70618488,
        107.58979057,
        99.91666667,
        92.68681316,
        85.90023005,
        79.55691735,
        73.65687504,
        68.20010313,
        63.18660163,
        58.57940003,
        54.34152786,
        50.4729851,
        46.97377176,
        43.84388784,
        41.08333333,
        38.69210825,
        36.67021258,
        35.01764633,
        33.73440951,
        32.82050209,
        32.2759241,
        31.97296144,
        31.78390001,
        31.70873982,
        31.74748086,
        31.90012315,
        32.16666667,
        32.54711142,
        33.04145742,
        33.64970465,
        34.37185312,
        35.20790282,
        36.15785376,
        37.26801653,
        38.58470172,
        40.10790933,
        41.83763935,
        43.7738918,
        45.91666667,
        48.26596395,
        50.82178366,
        53.58412578,
        56.55299032,
        59.72837728,
        63.11028667,
        67.02310605,
        71.79122301,
        77.41463755,
        83.89334968,
        91.22735938,
        99.41666667,
        108.46127153,
        118.36117398,
        129.11637401,
        140.72687162,
        153.19266681,
        166.51375958,
        181.49843978,
        198.95499726,
        218.88343202,
        241.28374407,
        266.15593339,
        293.5,
        323.31594389,
        355.60376506,
        390.36346351,
        427.59503924,
        467.29849225,
        509.47382255,
        552.31727381,
        594.02508973,
        634.59727031,
        674.03381555,
        712.33472545,
        749.5,
        785.52963921,
        820.42364308,
        854.18201161,
        886.80474479,
        918.29184263,
        948.64330513,
        976.28773216,
        999.65372359,
        1018.74127943,
        1033.55039966,
        1044.0810843,
        1050.33333333,
        1052.30714677,
        1050.00252461,
        1043.41946686,
        1032.5579735,
        1017.41804455,
        997.99967999,
        977.02114803,
        957.20071686,
        938.53838647,
        921.03415686,
        904.68802804,
        889.5,
        875.47007275,
        862.59824628,
        850.88452059,
        840.32889569,
        830.93137158,
        822.69194825,
        815.37241668,
        808.73456786,
        802.77840178,
        797.50391844,
        792.91111785,
        789.0,
        785.77056489,
        783.22281253,
        781.35674292,
        780.17235604,
        779.66965191,
        779.84863052,
        780.46888892,
        781.29002415,
        782.3120362,
        783.53492508,
        784.9586908,
        786.58333333,
        788.4088527,
        790.43524889,
        792.66252192,
        795.09067177,
        797.71969844,
        800.54960195,
        803.02721276,
        804.59936133,
        805.26604768,
        805.02727179,
        803.88303368,
        801.83333333,
        798.87817076,
        795.01754596,
        790.25145892,
        784.57990966,
        778.00289817,
        770.52042444,
        763.0159827,
        756.37306713,
        750.59167774,
        745.67181454,
        741.61347751,
        738.41666667,
        736.081382,
        734.60762352,
        733.99539122,
        734.24468509,
        735.35550515,
        737.32785139,
        739.61300218,
        741.6622359,
        743.47555253,
        745.0529521,
        746.39443459,
        747.5,
        748.36964834,
        749.0033796,
        749.40119378,
        749.56309089,
        749.48907093,
        749.17913389,
        749.28980052,
        750.47759157,
        752.74250705,
        756.08454694,
        760.50371126,
        766.0,
        772.57341316,
        780.22395074,
        788.95161274,
        798.75639917,
        809.63831001,
        821.59734528,
        834.56717618,
        848.48147392,
        863.34023851,
        879.14346994,
        895.89116821,
        913.58333333,
        932.2199653,
        951.8010641,
        972.32662975,
        993.79666225,
        1016.21116159,
        1039.57012777,
        1062.26316093,
        1082.67986119,
        1100.82022857,
        1116.68426305,
        1130.27196464,
        1141.58333333,
        1150.61836914,
        1157.37707205,
        1161.85944207,
        1164.06547919,
        1163.99518343,
        1161.64855477,
        1157.67561751,
        1152.72639595,
        1146.80089008,
        1139.89909992,
        1132.02102544,
        1123.16666667,
        1113.33602359,
        1102.5290962,
        1090.74588451,
        1077.98638852,
        1064.25060823,
        1049.53854363,
        1034.03498586,
        1017.92472607,
        1001.20776426,
        983.88410042,
        965.95373456,
        947.41666667,
        928.27289675,
        908.52242481,
        888.16525085,
        867.20137486,
        845.63079684,
        823.4535168,
        800.82280064,
        777.89191427,
        754.66085769,
        731.12963089,
        707.29823388,
        683.16666667,
        658.73492924,
        634.0030216,
        608.97094374,
        583.63869568,
        558.0062774,
        532.07368891,
        506.86154363,
        483.39045498,
        461.66042296,
        441.67144757,
        423.4235288,
        406.91666667,
        392.15086116,
        379.12611228,
        367.84242003,
        358.29978441,
        350.49820541,
        344.43768305,
        339.53803014,
        335.21905953,
        331.48077121,
        328.32316518,
        325.74624144,
        323.75,
        322.33444085,
        321.49956399,
        321.24536942,
        321.57185714,
        322.47902716,
        323.96687946,
        325.32925698,
        325.86000264,
        325.55911644,
        324.42659838,
        322.46244845,
        319.66666667,
        316.03925302,
        311.58020751,
        306.28953013,
        300.1672209,
        293.2132798,
        285.42770685,
        276.81050203,
        267.36166534,
        257.0811968,
        245.9690964,
        234.02536413,
    ]
)

vehicle_starting = np.array(
    [
        469.1171328,
        443.69548489,
        419.09959956,
        395.3285459,
        372.38106805,
        350.25383391,
        328.94375292,
        308.44819234,
        288.76128005,
        269.8762396,
        251.78508616,
        234.47653096,
        217.93708628,
        202.14904537,
        187.23364639,
        173.33538936,
        160.74133748,
        149.35005648,
        138.9484643,
        129.82635441,
        121.86598415,
        114.77473984,
        108.30635234,
        102.36660628,
        97.16858696,
        92.57561054,
        88.42882506,
        84.9448748,
        82.11069425,
        79.58333193,
        77.5308616,
        75.98326301,
        74.80147557,
        74.06382282,
        73.62570768,
        73.39617194,
        73.20922493,
        73.35049,
        73.77324004,
        74.70100345,
        76.11377869,
        78.1950941,
        80.86988539,
        84.1994823,
        88.34546937,
        93.11287441,
        98.37821869,
        104.4567326,
        111.30556965,
        118.90798256,
        127.38109469,
        136.71400294,
        147.19850848,
        158.96430308,
        172.74985618,
        188.67018516,
        206.53799311,
        226.19393513,
        247.48651269,
        270.21151144,
        294.82326303,
        320.9848637,
        348.15693932,
        377.26242984,
        408.67499197,
        443.6879152,
        482.90415449,
        526.40939645,
        575.4673876,
        630.5359741,
        691.99486404,
        759.30690206,
        831.84679481,
        909.53516619,
        992.46510547,
        1081.25324145,
        1173.23263127,
        1267.31224022,
        1364.385995,
        1462.72929862,
        1562.97276692,
        1663.41149118,
        1761.77046943,
        1856.24064997,
        1944.70763995,
        2025.39263178,
        2098.21347757,
        2164.03157446,
        2222.80669116,
        2273.48438739,
        2315.08911371,
        2346.38202128,
        2368.67843287,
        2382.08418138,
        2385.99931324,
        2380.52794821,
        2366.94203708,
        2345.79971447,
        2318.22555807,
        2284.72580097,
        2245.20222551,
        2203.06226955,
        2160.58956689,
        2118.46569656,
        2079.22750844,
        2041.41607667,
        2005.03172182,
        1971.0502901,
        1939.648819,
        1912.60281951,
        1890.38957103,
        1873.10769447,
        1859.78148484,
        1848.09949687,
        1838.82995101,
        1832.17746538,
        1828.16038744,
        1826.51808786,
        1826.21735968,
        1826.88007429,
        1828.39874915,
        1829.97937969,
        1831.9997007,
        1833.99369168,
        1835.93182816,
        1837.87919191,
        1839.59113463,
        1841.96620398,
        1844.7362667,
        1847.28073761,
        1849.24014321,
        1851.09931602,
        1853.19725001,
        1855.19382063,
        1856.53563205,
        1856.05883598,
        1854.94215668,
        1852.53783865,
        1850.37603606,
        1847.62877204,
        1843.30421379,
        1837.17846012,
        1829.8732786,
        1822.0078322,
        1813.28749445,
        1804.67059695,
        1795.46792375,
        1785.45796328,
        1775.88200036,
        1767.33748236,
        1759.81284511,
        1754.31933693,
        1749.66948686,
        1745.61678166,
        1743.22800768,
        1742.59776868,
        1743.94918182,
        1745.7116133,
        1749.05968815,
        1752.41008736,
        1756.83030659,
        1760.49176821,
        1763.90883261,
        1766.99951358,
        1769.39584602,
        1770.97625561,
        1773.56852645,
        1776.38038594,
        1778.85975329,
        1782.2013791,
        1785.37939945,
        1789.42351973,
        1794.34570127,
        1800.64273285,
        1808.5826721,
        1819.02606153,
        1831.28830973,
        1845.65899969,
        1863.24776831,
        1883.72164382,
        1907.44118734,
        1933.05725526,
        1960.80428054,
        1987.41740859,
        2014.4712567,
        2042.39166814,
        2072.5155976,
        2106.33447882,
        2143.0250319,
        2180.90762014,
        2219.30359959,
        2257.55233811,
        2295.50058022,
        2333.36668476,
        2368.58085581,
        2399.09334357,
        2424.29216816,
        2445.45334539,
        2462.6604712,
        2475.8334986,
        2484.59287628,
        2487.94201616,
        2486.32796115,
        2479.98421509,
        2468.80993952,
        2453.25160052,
        2433.6751749,
        2410.11160915,
        2385.05647488,
        2360.40997268,
        2337.06571142,
        2315.82406424,
        2295.76955681,
        2276.06490635,
        2257.9385966,
        2241.28583533,
        2226.28249517,
        2212.01558544,
        2196.48011568,
        2179.73254121,
        2162.87568558,
        2146.31768362,
        2129.42113472,
        2111.16633946,
        2090.90368935,
        2067.58560761,
        2041.11327124,
        2012.64983009,
        1981.63352362,
        1947.75371966,
        1910.31014311,
        1868.9113914,
        1823.77345504,
        1776.3958802,
        1726.31116668,
        1674.19311091,
        1620.77185539,
        1565.16114384,
        1507.89083843,
        1448.97780561,
        1390.15982787,
        1331.60992067,
        1273.98666373,
        1217.4900917,
        1162.83665329,
        1110.51401921,
        1061.01560972,
        1015.06383647,
        972.2161952,
        932.84820362,
        897.53349296,
        867.24168728,
        840.53357721,
        817.12167681,
        796.76957585,
        779.17538322,
        764.10398194,
        751.85304426,
        741.9860572,
        734.07151811,
        728.20817781,
        723.34565396,
        719.66960917,
        717.20425799,
        715.89884563,
        715.49402594,
        715.64805251,
        716.40935477,
        717.33242379,
        718.40851324,
        719.31175849,
        719.26786956,
        717.7663086,
        715.35734916,
        711.54710803,
        705.86680218,
        697.68578456,
        688.20500293,
        677.37969251,
        665.15237417,
        651.44865567,
        636.17305282,
        619.20255347,
        600.38112308,
        579.50875306,
        556.3263212,
        530.50311514,
        501.61506706,
    ]
)

euro_class_split = {
    "euro_0": 0.059,
    "euro_1": 0.012,
    "euro_2": 0.034,
    "euro_3": 0.054,
    "euro_4": 0.198,
    "euro_5": 0.176,
    "euro_6": 0.467,
}

# Emissions nox per car per km
euro_class_emission = {
    "euro_0": 0.210584391986267347,
    "euro_1": 0.2174573179869368,
    "euro_2": 0.24014520073869067,
    "euro_3": 0.24723923486567853,
    "euro_4": 0.1355550834386541,
    "euro_5": 0.09955851060544411,
    "euro_6": 0.06824599009858062,
}


def _ts_solve(ts: np.ndarray) -> np.ndarray:
    """Solve traffic with iterative method.

    Computes steady-state circulating traffic from an inflow time series by
    iterating a simple feedback loop until convergence (50 iterations).

    Parameters
    ----------
    ts:
        Inflow timeseries.  Shape ``(T,)`` for a single scenario or
        ``(S, T)`` for an ensemble of *S* samples.

    Returns
    -------
    np.ndarray
        Circulating traffic, same shape as *ts*.
    """
    tot_traffic = 2_200_368.245_435_709  # TODO: should not be a constant value
    series = ts
    for _ in range(50):  # TODO: decide when to finish based on convergence
        mu = 1.0 + 3.0 * series.sum(axis=-1, keepdims=True) / tot_traffic
        alfa = (mu - 1.0) / mu
        series = ts + np.roll(series, 1, axis=-1) * alfa
    return series


# MODEL DEFINITION


class Model:
    """Model for the Bologna mobility example."""

    def __init__(self):
        self.TS = TimeseriesIndex(
            "time range",
            np.array(
                [
                    (t - pd.Timestamp("00:00:00")).total_seconds()
                    for t in pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")
                ]
            ),
        )

        # Flow variables
        self.TS_inflow = TimeseriesIndex("inflow", vehicle_inflow)
        self.TS_starting = TimeseriesIndex("staring", vehicle_starting)

        # Indexes
        self.I_P_start_time = Index("start time", (pd.Timestamp("07:30:00") - pd.Timestamp("00:00:00")).total_seconds())
        self.I_P_end_time = Index("end time", (pd.Timestamp("19:30:00") - pd.Timestamp("00:00:00")).total_seconds())

        self.I_P_cost = [Index(f"cost euro {e}", 5.00 - e * 0.25) for e in range(7)]

        self.I_P_fraction_exempted = Index("exempted vehicles %", 0.15)

        self.I_B_p50_cost = UniformDistIndex("cost 50% threshold", loc=4.00, scale=7.00)
        self.I_B_p50_anticipating = Index("anticipation 50% likelihood", 0.5)
        self.I_B_p50_anticipation = Index("anticipation distribution 50% threshold", 0.25)
        self.I_B_p50_postponing = Index("postponement 50% likelihood", 0.8)
        self.I_B_p50_postponement = Index("postponement distribution 50% threshold", 0.50)
        self.I_B_starting_modified_factor = Index("starting modified factor", 1.00)

        self.I_avg_cost = Index(
            "average cost",
            self.I_P_cost[0] * euro_class_split["euro_0"]
            + self.I_P_cost[1] * euro_class_split["euro_1"]
            + self.I_P_cost[2] * euro_class_split["euro_2"]
            + self.I_P_cost[3] * euro_class_split["euro_3"]
            + self.I_P_cost[4] * euro_class_split["euro_4"]
            + self.I_P_cost[5] * euro_class_split["euro_5"]
            + self.I_P_cost[6] * euro_class_split["euro_6"],
        )

        self.I_fraction_rigid_euro = [
            Index(
                f"rigid vehicles euro_{e} %",
                (1 - self.I_P_fraction_exempted) * (np.e ** (self.I_P_cost[e] / self.I_B_p50_cost * np.log(0.5))),
            )
            for e in range(7)
        ]

        self.I_fraction_rigid = Index(
            "rigid vehicles %",
            self.I_fraction_rigid_euro[0] * euro_class_split["euro_0"]
            + self.I_fraction_rigid_euro[1] * euro_class_split["euro_1"]
            + self.I_fraction_rigid_euro[2] * euro_class_split["euro_2"]
            + self.I_fraction_rigid_euro[3] * euro_class_split["euro_3"]
            + self.I_fraction_rigid_euro[4] * euro_class_split["euro_4"]
            + self.I_fraction_rigid_euro[5] * euro_class_split["euro_5"]
            + self.I_fraction_rigid_euro[6] * euro_class_split["euro_6"],
        )

        self.I_modified_euro_class_split = [
            Index(
                f"modified split euro_{e} %",
                euro_class_split[f"euro_{e}"]
                * (self.I_P_fraction_exempted + self.I_fraction_rigid_euro[e])
                / (self.I_P_fraction_exempted + self.I_fraction_rigid),
            )
            for e in range(7)
        ]

        self.I_delta_from_start = TimeseriesIndex(
            "delta time from start",
            Piecewise(
                ((self.TS - self.I_P_start_time) / pd.Timedelta("1h").total_seconds(), self.TS >= self.I_P_start_time),
                (np.inf, True),
            ),
        )

        self.I_fraction_anticipating = TimeseriesIndex(
            "anticipating vehicles %",
            np.e ** (self.I_delta_from_start / self.I_B_p50_anticipating * np.log(0.5))
            * (1 - self.I_P_fraction_exempted - self.I_fraction_rigid),
        )

        self.I_number_anticipating = TimeseriesIndex(
            "anticipating vehicles", self.I_fraction_anticipating * self.TS_inflow
        )

        self.I_delta_to_end = TimeseriesIndex(
            "delta time to end",
            Piecewise(
                ((self.I_P_end_time - self.TS) / pd.Timedelta("1h").total_seconds(), self.TS <= self.I_P_end_time),
                (np.inf, True),
            ),
        )

        self.I_fraction_postponing = TimeseriesIndex(
            "postponing vehicles %",
            np.e ** (self.I_delta_to_end / self.I_B_p50_postponing * np.log(0.5))
            * (1 - self.I_P_fraction_exempted - self.I_fraction_rigid),
        )

        self.I_number_postponing = TimeseriesIndex("postponing vehicles", self.I_fraction_postponing * self.TS_inflow)

        self.I_total_anticipating = Index("total anticipating vehicles", self.I_number_anticipating.sum())

        self.I_total_postponing = Index("total postponing vehicles", self.I_number_postponing.sum())

        self.I_delta_before_start = TimeseriesIndex(
            "delta time before start",
            Piecewise(
                ((self.I_P_start_time - self.TS) / pd.Timedelta("1h").total_seconds(), self.TS < self.I_P_start_time),
                (np.inf, True),
            ),
        )
        self.I_number_anticipated = TimeseriesIndex(
            "anticipated vehicles",
            np.e ** (self.I_delta_before_start / self.I_B_p50_anticipation * np.log(0.5))
            / self.I_B_p50_anticipation
            * np.log(2)
            / 12
            * self.I_total_anticipating,
        )

        self.I_delta_after_end = TimeseriesIndex(
            "delta time after end",
            Piecewise(
                ((self.TS - self.I_P_end_time) / pd.Timedelta("1h").total_seconds(), self.TS > self.I_P_end_time),
                (np.inf, True),
            ),
        )
        self.I_number_postponed = TimeseriesIndex(
            "postponed vehicles",
            np.e ** (self.I_delta_after_end / self.I_B_p50_postponement * np.log(0.5))
            / self.I_B_p50_postponement
            * np.log(2)
            / 12
            * self.I_total_postponing,
        )

        self.I_number_shifted = TimeseriesIndex("shifted vehicles", self.I_number_anticipated + self.I_number_postponed)

        self.I_modified_inflow = Index(
            "modified vehicle inflow",
            Piecewise(
                (
                    (self.I_P_fraction_exempted + self.I_fraction_rigid) * self.TS_inflow,
                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time),
                ),
                (self.TS_inflow + self.I_number_shifted, True),
            ),
        )

        self.I_total_base_inflow = Index("total base vehicle flow", self.TS_inflow.sum())
        self.I_total_modified_inflow = Index("total modified vehicle inflow", self.I_modified_inflow.sum())
        self.I_number_paying = Index(
            "paying vehicles",
            Piecewise(
                (
                    self.I_fraction_rigid * self.TS_inflow,
                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time),
                ),
                (0, True),
            ),
        )

        self.I_total_paying = Index("total vehicles paying", self.I_number_paying.sum())

        self.I_modified_starting = Index(
            "modified starting", self.TS_starting + self.I_modified_inflow * (self.I_B_starting_modified_factor - 1)
        )

        # TODO: flat rates for all AV
        self.I_inflow_ratio = Index(
            "ratio between modified flow and base flow", self.TS_inflow / self.I_modified_inflow
        )

        self.I_starting_ratio = Index(
            "ratio between modified starting and base starting", self.TS_starting / self.I_modified_starting
        )

        # TODO: fix, compute real value!
        self.I_total_payed = Index("total payed fees", self.I_total_paying * self.I_avg_cost)

        self.I_total_anticipated = Index("total vehicles anticipated", self.I_number_anticipated.sum())

        self.I_total_postponed = Index("total vehicles postponed", self.I_number_postponed.sum())

        self.I_total_shifted = Index("total vehicles shifted", self.I_total_anticipated + self.I_total_postponed)

        self.I_traffic = TimeseriesIndex(
            "reference traffic", graph.function_call("ts_solve", self.TS_inflow + self.TS_starting)
        )

        self.I_modified_traffic = TimeseriesIndex(
            "modified traffic", graph.function_call("ts_solve", self.I_modified_inflow + self.I_modified_starting)
        )

        self.I_traffic_ratio = Index(
            "ratio between modified traffic and base traffic", self.I_traffic / self.I_modified_traffic
        )

        self.I_average_emissions = Index(
            "average emissions (per vehicle, per km)",
            euro_class_emission["euro_0"] * euro_class_split["euro_0"]
            + euro_class_emission["euro_1"] * euro_class_split["euro_1"]
            + euro_class_emission["euro_2"] * euro_class_split["euro_2"]
            + euro_class_emission["euro_3"] * euro_class_split["euro_3"]
            + euro_class_emission["euro_4"] * euro_class_split["euro_4"]
            + euro_class_emission["euro_5"] * euro_class_split["euro_5"]
            + euro_class_emission["euro_6"] * euro_class_split["euro_6"],
        )

        self.I_modified_average_emissions = Index(
            "modified average emissions (per vehicle, per km)",
            euro_class_emission["euro_0"] * self.I_modified_euro_class_split[0]
            + euro_class_emission["euro_1"] * self.I_modified_euro_class_split[1]
            + euro_class_emission["euro_2"] * self.I_modified_euro_class_split[2]
            + euro_class_emission["euro_3"] * self.I_modified_euro_class_split[3]
            + euro_class_emission["euro_4"] * self.I_modified_euro_class_split[4]
            + euro_class_emission["euro_5"] * self.I_modified_euro_class_split[5]
            + euro_class_emission["euro_6"] * self.I_modified_euro_class_split[6],
        )

        # TODO: improve - at the moment, the conversion factor is 2,5 km per 5 minutes
        self.I_emissions = TimeseriesIndex("emissions", 2.5 * self.I_average_emissions * self.I_traffic)

        # I_modified_emissions = Index('modified emissions', 2.5 * I_modified_average_emissions * I_modified_traffic)
        #
        # TODO: The average emissions is probably different outside regulated hours
        #  (shifted cars' emissions are probably proportional to shifted cars' euro level mix)
        #
        self.I_modified_emissions = Index(
            "modified emissions",
            Piecewise(
                (
                    2.5 * self.I_modified_average_emissions * self.I_modified_traffic,
                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time),
                ),
                (2.5 * self.I_average_emissions * self.I_modified_traffic, True),
            ),
        )

        self.I_total_emissions = Index("total emissions", self.I_emissions.sum())

        self.I_total_modified_emissions = Index("total modified emissions", self.I_modified_emissions.sum())

        self.indexes = [
            self.TS,
            self.TS_inflow,
            self.TS_starting,
            self.I_P_start_time,
            self.I_P_end_time,
            *self.I_P_cost,
            self.I_P_fraction_exempted,
            self.I_B_p50_cost,
            self.I_B_p50_anticipating,
            self.I_B_p50_anticipation,
            self.I_B_p50_postponing,
            self.I_B_p50_postponement,
            self.I_B_starting_modified_factor,
            self.I_avg_cost,
            *self.I_fraction_rigid_euro,
            self.I_fraction_rigid,
            *self.I_modified_euro_class_split,
            self.I_delta_from_start,
            self.I_fraction_anticipating,
            self.I_number_anticipating,
            self.I_delta_to_end,
            self.I_fraction_postponing,
            self.I_number_postponing,
            self.I_total_anticipating,
            self.I_total_postponing,
            self.I_delta_before_start,
            self.I_number_anticipated,
            self.I_delta_after_end,
            self.I_number_postponed,
            self.I_number_shifted,
            self.I_modified_inflow,
            self.I_modified_starting,
            self.I_inflow_ratio,
            self.I_starting_ratio,
            self.I_total_base_inflow,
            self.I_total_modified_inflow,
            self.I_number_paying,
            self.I_total_paying,
            self.I_total_payed,
            self.I_total_anticipated,
            self.I_total_postponed,
            self.I_total_shifted,
            self.I_traffic,
            self.I_modified_traffic,
            self.I_traffic_ratio,
            self.I_average_emissions,
            self.I_modified_average_emissions,
            self.I_emissions,
            self.I_modified_emissions,
            self.I_total_emissions,
            self.I_total_modified_emissions,
        ]

    def evaluate(self, size=1):
        """Evaluate ensemble model."""
        from civic_digital_twins.dt_model.engine.frontend import linearize
        from civic_digital_twins.dt_model.engine.numpybackend import executor

        # Pre-populate state with sampled values for distribution-backed placeholders.
        initial_values = {}
        for index in self.indexes:
            if hasattr(index.value, "rvs"):
                initial_values[index.node] = np.expand_dims(np.array(index.value.rvs(size=size)), axis=1)

        state = executor.State(
            initial_values,
            # compileflags.defaults | compileflags.TRACE,
            functions={"ts_solve": executor.LambdaAdapter(_ts_solve)},
        )

        # Evaluate the full graph in topological order.
        plan = linearize.forest(*[idx.node for idx in self.indexes])
        executor.evaluate_nodes(state, *plan)

        # Build the subs dictionary with appropriate shape conventions.
        subs = {}
        for index in self.indexes:
            if isinstance(index, TimeseriesIndex):
                if len(state.values[index.node].shape) == 2:
                    subs[index] = state.values[index.node]
                else:
                    subs[index] = np.expand_dims(state.values[index.node], axis=0)
            elif isinstance(index, Index):
                val = state.values[index.node]
                if len(val.shape) == 0:
                    # Bare scalar (e.g. constant node) — replicate across ensemble
                    subs[index] = np.expand_dims(np.array([float(val)] * size), axis=1)
                elif len(val.shape) == 1:
                    # 1-D: no ensemble dimension yet — add it as axis 0
                    subs[index] = np.expand_dims(val, axis=0)
                else:
                    subs[index] = val
            else:
                raise (ValueError, f"{index.name} is not a TimeseriesIndex or Index")
            if len(subs[index].shape) != 2:
                raise (ValueError, f"wrong dimension for subs[{index.name}]: {subs[index].shape}")

        return subs


def distribution(field, size=10000, num=100):
    """Compute the field distribution for graphical display."""
    xx, yy = np.meshgrid(np.linspace(0, size, num + 1), range(field.shape[1]))
    zz = stats.poisson(mu=np.expand_dims(field, axis=2)).cdf(np.expand_dims(xx, axis=0))
    return zz.mean(axis=0)


field_color = (165 / 256, 15 / 256, 21 / 256)
delta = 0.5
field_light_color = (
    (field_color[0] + delta) / (1 + delta),
    (field_color[1] + delta) / (1 + delta),
    (field_color[2] + delta) / (1 + delta),
)

field_colormap = LinearSegmentedColormap.from_list(
    "mid_red_bar", colors=["white", field_light_color, field_color, field_light_color, "white"], N=100
)


def plot_field_graph(
    field, horizontal_label, vertical_label, vertical_size=None, vertical_formatter=None, reference_line=None
):
    """Generate plot figure."""
    if vertical_size is None:
        vertical_size = roundup(np.max(field))
    dist = distribution(field, vertical_size, 100)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    pcm = ax.pcolormesh(
        pd.date_range(start="00:00:00", periods=12 * 24, freq="5min"),
        np.linspace(0, vertical_size, 100 + 1),
        dist.T,
        cmap=field_colormap,
        vmin=0.0,
        vmax=1.0,
    )
    if reference_line is not None:
        ax.plot(
            pd.date_range(start="00:00:00", periods=12 * 24, freq="5min"),
            reference_line,
            "--",
            linewidth=1,
            color="black",
            label="Riferimento",
        )
    ax.plot(
        pd.date_range(start="00:00:00", periods=12 * 24, freq="5min"),
        field.mean(axis=0),
        linewidth=1,
        color="black",
        label="Modificato (mediana)",
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_ticks([0.00, 0.25, 0.50, 0.75, 1.00])
    cbar.set_ticklabels([f"{x}%" for x in [0, 25, 50, 75, 100]])
    ax.set_ylim([0, vertical_size])
    if vertical_formatter is not None:
        ax.yaxis.set_major_formatter(vertical_formatter)
    ax.set_ylabel(vertical_label)
    fig.tight_layout()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    ax.set_xlabel(horizontal_label)
    ax.legend(loc="upper right")
    return fig


def compute_kpis(m, evals):
    """Compute the KPIs for the mobility example."""
    return {
        "Base inflow [veh/day]": int(evals[m.I_total_base_inflow].mean()),
        "Modified inflow [veh/day]": int(evals[m.I_total_modified_inflow].mean()),
        "Shifted inflow [veh/day]": int(evals[m.I_total_shifted].mean()),
        "Paying inflow [veh/day]": int(evals[m.I_total_paying].mean()) if evals[m.I_avg_cost].mean() > 0 else 0,
        "Collected fees [€/day]": int(evals[m.I_total_payed].mean()),
        "Emissions [NOx gr/day]": int(evals[m.I_total_modified_emissions].mean()),
        "Modified emissions [NOx gr/day]": int(evals[m.I_total_emissions].mean())
        - int(evals[m.I_total_modified_emissions].mean()),
    }


def roundup(val):
    """Compute a rounded-up approximation of `val`."""
    v = val * 1.4
    s = math.floor(math.log10(v * 1.3))
    return round(v / 10**s) * 10**s


if __name__ == "__main__":
    m = Model()

    subs = m.evaluate(20)

    plot_field_graph(
        subs[m.I_modified_inflow],
        horizontal_label="Time",
        vertical_label="Flow (vehicles/hour)",
        vertical_size=1600,
        vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
        reference_line=subs[m.TS_inflow][0],
    )
    plt.show()
    plot_field_graph(
        subs[m.I_modified_traffic],
        horizontal_label="Time",
        vertical_label="Traffic (circulating vehicles)",
        vertical_size=15000,
        reference_line=subs[m.I_traffic][0],
    )
    plt.show()
    plot_field_graph(
        subs[m.I_modified_emissions],
        horizontal_label="Time",
        vertical_label="Emissions (NOx gr/h)",
        vertical_size=4000,
        vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
        reference_line=subs[m.I_emissions][0],
    )
    plt.show()
    for k, v in compute_kpis(m, subs).items():
        print(f"{k} - {v:,}")
