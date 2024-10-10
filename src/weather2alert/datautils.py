import pandas as pd

WESTERN_STATES = [
    "AZ",
    "CA",
    "CO",
    "ID",
    "MT",
    "NM",
    "NV",
    "OR",
    "WA",
    "ND",
    "SD",
    "NE",
    "KS",
]  # ND, SD, NE and KS together only add 10 counties to the cold western group

SOUTHERN_STATES = [
    "TX",
    "OK",
    "AR",
    "LA",
    "MS",
    "AL",
    "GA",
    "FL",
    "TN",
    "KY",
    "SC",
    "NC",
    "VA",
    "WV",
    "VA",
    "MD",
    "DE",
    "NM",
    "AZ",
    "CA",  # including CA in South based on the specific counties in our final 30
]

FIPS2STATE = {
    "01": "AL",
    "02": "AK",
    "03": "AZ",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
    "72": "PR",
    "60": "AS",
    "66": "GU",
    "69": "MP",
    "78": "VI",
}


def get_similar_counties(fips: str, confounders: pd.DataFrame):
    """Returns a dictionary that assigns similar counties to each county based
    on climate zones
    """

    confounders = confounders.copy().set_index("fips")
    confounders["state"] = confounders.index.str[:2].map(FIPS2STATE)
    confounders["western"] = confounders["state"].isin(WESTERN_STATES)

    # now replace Cold with Cold-West or Cold-East in column ba_zone
    confounders["ba_zone"] = confounders.apply(
        lambda x: (
            "Cold-West"
            if x["western"]
            else "Cold-East" if x["ba_zone"] == "Cold" else x["ba_zone"]
        ),
        axis=1,
    )

    # get all similar
    county_zone = confounders.loc[fips].ba_zone
    similar_counties = confounders[confounders.ba_zone == county_zone].index.tolist()

    return similar_counties
