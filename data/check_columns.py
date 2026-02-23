import pandas as pd

aws = pd.read_excel("data/aws_test-focus-00001.snappy_transformed.xls")
azure = pd.read_excel("data/focusazure_anon_transformed.xls")

print("AWS Columns:")
print(aws.columns)

print("\nAzure Columns:")
print(azure.columns)