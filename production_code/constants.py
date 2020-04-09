from datetime import datetime

__non_pow_start_date = "01-01-2007"
__non_pow_end_date = "30-06-2009"

__pow_start_date = "07-07-2007"
__pow_end_date = "13-03-2009"

NON_POW_START_DATE = datetime.strptime(__non_pow_start_date, "%d-%m-%Y")
NON_POW_END_DATE = datetime.strptime(__non_pow_end_date, "%d-%m-%Y")

POW_START_DATE = datetime.strptime(__pow_start_date, "%d-%m-%Y")
POW_END_DATE = datetime.strptime(__pow_end_date, "%d-%m-%Y")
