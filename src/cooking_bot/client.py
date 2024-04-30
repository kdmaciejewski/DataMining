from dotenv import load_dotenv
import os
from .data_formats import *
from opensearchpy import OpenSearch

assert load_dotenv()

INDEX_NAME = os.environ["NOVA_SEARCH_US"]

# Create the client with SSL/TLS enabled, but hostname verification disabled.
CLIENT = OpenSearch(
    hosts=[
        {"host": os.environ["NOVA_SEARCH_HOST"], "port": os.environ["NOVA_SEARCH_PORT"]}
    ],
    http_compress=True,  # enables gzip compression for request bodies
    http_auth=(os.environ["NOVA_SEARCH_US"], os.environ["NOVA_SEARCH_PW"]),
    url_prefix="opensearch",
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

print("asserting index exists")

assert CLIENT.indices.exists(INDEX_NAME)

print("Index exists!")


resp = CLIENT.indices.open(index = INDEX_NAME)
assert resp["acknowledged"], "Index did not open"