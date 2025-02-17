# from exe_guard import BRDRAG

# brdrag = BRDRAG()

# # Define multiple files in an array
# files = ["BRD.docx", "S4HC Digital Discovery Assessment Results.pdf"]

# # Define multiple queries in an array
# queries = [
#     "What are the table of contents?",
#     "Provide a summary of the document?",
#     "What is the key takeaway from the document?"
# ]

# # Get responses for all queries using the same vector store built from the files
# responses = brdrag.getResponse(files, queries)

# # Print each response with its associated document annotation
# for query, response in responses.items():
#     print(f"Response for query: '{query}'")
#     print(response)
#     print("=" * 80)








from exe_guard import BRDRAG

brdrag = BRDRAG()

# Define multiple files in an array
files = ["BRD.docx", "S4HC Digital Discovery Assessment Results.pdf"]

# Define multiple queries. Notice the first query includes "#1#"
queries = [
    "What are the table of contents? #1#",
    "Provide a summary of the document?",
    "What is the key takeaway from the document? #2#"
]

# Get responses for all queries using the vector store built from the files
responses = brdrag.getResponse(files, queries)

# Print each response
for query, response in responses.items():
    print(f"Response for query: '{query}'")
    print(response)
    print("=" * 80)
