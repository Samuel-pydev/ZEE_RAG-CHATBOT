vectorstore = None

def get_vectorstore():
    return vectorstore

def set_vectorestore(vs):
    global vectorstore
    vectorstore = vs