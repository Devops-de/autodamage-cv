"""
Regional Prevailing Wage and Sales Tax Matrix.
Reflects standardized US collision repair labor rate variance across geographic markets.
"""

from pydantic import BaseModel


class RegionalMarket(BaseModel):
    market_code: str
    region_name: str
    labor_rate_multiplier: float
    sales_tax_rate: float
    description: str


REGIONAL_MARKETS: dict[str, RegionalMarket] = {
    "PACIFIC_METRO": RegionalMarket(
        market_code="PACIFIC_METRO",
        region_name="Pacific Metro (CA / WA / OR)",
        labor_rate_multiplier=1.40,  # ~$91/hr body & paint labor
        sales_tax_rate=0.085,  # 8.5% average sales tax
        description="High-cost Pacific coastal metropolitan area",
    ),
    "NORTHEAST_METRO": RegionalMarket(
        market_code="NORTHEAST_METRO",
        region_name="Northeast Metro (NY / MA / NJ)",
        labor_rate_multiplier=1.35,  # ~$88/hr body & paint labor
        sales_tax_rate=0.0825,  # 8.25% average sales tax
        description="Northeastern urban collision repair market",
    ),
    "SOUTHEAST_CENTRAL": RegionalMarket(
        market_code="SOUTHEAST_CENTRAL",
        region_name="South / Southeast (TX / FL / GA)",
        labor_rate_multiplier=1.15,  # ~$75/hr body & paint labor
        sales_tax_rate=0.075,  # 7.5% average sales tax
        description="Southern automotive repair market",
    ),
    "MIDWEST_MOUNTAIN": RegionalMarket(
        market_code="MIDWEST_MOUNTAIN",
        region_name="Midwest / Mountain (IL / OH / CO)",
        labor_rate_multiplier=1.05,  # ~$68/hr body & paint labor
        sales_tax_rate=0.070,  # 7.0% average sales tax
        description="Midwestern and Rocky Mountain market",
    ),
    "NATIONAL_BASELINE": RegionalMarket(
        market_code="NATIONAL_BASELINE",
        region_name="US National Baseline",
        labor_rate_multiplier=1.00,  # $65/hr body & paint labor baseline
        sales_tax_rate=0.070,  # 7.0% baseline sales tax
        description="Standard US collision industry baseline rate",
    ),
}


def lookup_regional_market(zip_code: str | None = None) -> RegionalMarket:
    """
    Resolves the prevailing wage tier based on 5-digit US ZIP code prefix,
    falling back to standard national baseline rates if not provided or unknown.
    """
    if not zip_code or not isinstance(zip_code, str):
        return REGIONAL_MARKETS["NATIONAL_BASELINE"]

    clean_zip = zip_code.strip()
    if len(clean_zip) < 3 or not clean_zip[:3].isdigit():
        return REGIONAL_MARKETS["NATIONAL_BASELINE"]

    prefix = clean_zip[0]

    # ZIP Prefix 9: CA, OR, WA, AK, HI
    if prefix == "9":
        return REGIONAL_MARKETS["PACIFIC_METRO"]
    # ZIP Prefix 0 or 1: MA, NJ, NY, PA, CT
    if prefix in ("0", "1"):
        return REGIONAL_MARKETS["NORTHEAST_METRO"]
    # ZIP Prefix 3 or 7: FL, GA, TX, LA
    if prefix in ("3", "7"):
        return REGIONAL_MARKETS["SOUTHEAST_CENTRAL"]
    # ZIP Prefix 4, 5, 6, 8: OH, IN, MI, IA, MN, IL, MO, CO, AZ, UT
    if prefix in ("4", "5", "6", "8"):
        return REGIONAL_MARKETS["MIDWEST_MOUNTAIN"]
    return REGIONAL_MARKETS["NATIONAL_BASELINE"]
