from datasets import load_dataset
# noinspection PyPackageRequirements
from haystack.nodes import PreProcessor, CsvTextConverter
# noinspection PyPackageRequirements
from haystack.utils import convert_files_to_docs
from huggingface_hub import HfApi

dataset = load_dataset("ojigs/bgNlp-QA")
doc = convert_files_to_docs(dir_path="/data")

converter = CsvTextConverter(remove_numeric_tables=True, valid_languages=["en"])
doc_txt = converter.convert(file_path="data/text_segments.csv", meta=None)[0]

api = HfApi()
# api.upload_file()

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=1,
    split_overlap=0,
    split_respect_sentence_boundary=False
)

processed_doc = preprocessor.process(doc_txt)

print(doc)
print(f"n_files_input: {len(doc)}\nn_docs_output: {len(processed_doc)}")

# processed_dataset.push_to_hub("ojigs/bgNlp-QA")
