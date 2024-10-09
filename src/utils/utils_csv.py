import csv

def read_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        leitor = csv.reader(csvfile)
        for linha in leitor:
            yield linha

def line_to_str(csv_line_list):
    return ",".join([word for word in csv_line_list])