import uvicorn
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
