
# from executor import BRDRAG

# brdrag = BRDRAG()
# response_DDA = brdrag.getResponse(["BRD.docx"], "What are the table of contents?")
# print("Response for DDA: ")
# print(response_DDA)

#response_MOM = brdrag.getResponse(["2023.12.18_Tamkeen_Feed_Source to Pay MoM_V1.0.docx"], "What are the action items?")
#print("Response for MOM: ")
#print(response_MOM)


#response_KDS = brdrag.getResponse(["MATERIAL MANAGEMENT - KDS UPDATED.xlsx"], "What are different KDS List?")
#print("Response for KDS: ")
#print(response_DDA)


from executor import BRDRAG

brdrag = BRDRAG()

# Define multiple files in an array
files = ["BRD.docx", "S4HC Digital Discovery Assessment Results.pdf"]

# Define multiple queries in an array
queries = [
    "What are the table of contents?",
    "Provide a summary of the document?",
    "What is the key takeaway from the document?"
]

# Get responses for all queries using the same vector store built from the files
responses = brdrag.getResponse(files, queries)

# Print each response with its associated document annotation
for query, response in responses.items():
    print(f"Response for query: '{query}'")
    print(response)
    print("=" * 80)
