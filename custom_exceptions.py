class SVDError(Exception):
    def __init__(self, message="SVD does not yield original matrix."):
        super(SVDError, self).__init__(message)