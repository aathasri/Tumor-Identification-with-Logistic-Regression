import GEOparse

# Download GSE15852
gse = GEOparse.get_GEO("GSE15852", destdir="./GSE15852_raw")

import pandas as pd

# gse.pivot_samples() creates a DataFrame with probes as rows and samples as columns
expr_data = gse.pivot_samples("VALUE")  # 'VALUE' is the raw GEO intensity

# Save probe-level data to TSV
expr_data.to_csv("GSE15852_probe_data.tsv", sep="\t")


# Create metadata DataFrame
metadata = pd.DataFrame({
    "SampleID": [gsm.name for gsm in gse.gsms.values()],
    "Title": [gsm.metadata.get("title", [""])[0] for gsm in gse.gsms.values()],
    "Source": [gsm.metadata.get("source_name_ch1", [""])[0] for gsm in gse.gsms.values()],
    "Tissue": [gsm.metadata.get("characteristics_ch1", [""])[0] for gsm in gse.gsms.values()],
    "Histopathology": [gsm.metadata.get("characteristics_ch1", [""])[1] for gsm in gse.gsms.values()],
    "Grade": [gsm.metadata.get("characteristics_ch1", [""])[2] for gsm in gse.gsms.values()],
    "Age": [gsm.metadata.get("characteristics_ch1", [""])[3] for gsm in gse.gsms.values()],
    "Race": [gsm.metadata.get("characteristics_ch1", [""])[4] for gsm in gse.gsms.values()],
})

print(gsm.metadata.get("characteristics_ch1", [""]) for gsm in gse.gsms.values())

# Save metadata to TSV
metadata.to_csv("GSE15852_metadata.tsv", sep="\t", index=False)