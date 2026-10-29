CT4101 Bike Sharing

Regression project: Predict hourly Capital Bikeshare rentals in Washington D.C. in 2011 and 2012 from calendar and weather data

Dataset:
Name: Bike Sharing Dataset
Source: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
File used: `hour.csv` 17,379 hourly records

Setup
Python 3.12.

```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```
## Run the baseline (Milestone 2)

Place `hour.csv` in `data/`. 
Then from the project root with the virtual environment active,
.\.venv\Scripts\activate.bat
python src/baseline.py

The script prints a data audit, the time-based split (train before the 1st of October 2012, test from that date onward)
and MAE / RMSE / R² for an hour-of-day mean baseline and a global-mean reference. 
The test block is not used for model choice.