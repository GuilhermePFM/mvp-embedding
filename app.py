
from flask import redirect
from schemas.embedding_schema import *

from apis.embedding import *
from config import *

@app.get('/', tags=[home_tag])
def home():
    """Redirects to /openapi to choose the doc style
    """
    return redirect('/openapi')


if __name__ == '__main__':
    app.run()