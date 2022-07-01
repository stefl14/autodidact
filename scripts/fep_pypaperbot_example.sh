# Get papers that relate to Karl Friston's Free Energy Principle from 2010 onwards and put them in the downloads directory.
# Refresh.
rm -rf downloads/*
python -m PyPaperBot --query="Free Energy Principle" --scholar-pages=3  --min-year=2010 --dwn-dir="downloads" --scihub-mirror="https://sci-hub.yncjkj.com/"
# Move results.csv to ../pg_data/results.csv.
mv downloads/results.csv ../pg_data/results.csv