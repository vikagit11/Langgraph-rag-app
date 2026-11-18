# Hybrid and Multi-Stage Queries (Qdrant)

_Available as of v1.10.0_

You can search the same data in several ways: with dense vectors for semantic search, with sparse vectors from learned models like SPLADE++, and with BM25 full-text search on text payload fields that have a full-text index. Often, the best results come from combining these queries, or from searching in more than one stage, for example, fetching candidates with a fast query and re-scoring them with a more accurate one.

Qdrant supports this through the Query API. Its main building block is the `prefetch` parameter, which lets a query run sub-requests. Whenever a query has at least one prefetch, Qdrant:

1. Runs the prefetch queries.
2. Applies the main query to the results of the prefetches.

Additionally, prefetches can have prefetches themselves, so you can have nested prefetches.

Using `offset` parameter only affects the main query. This means that the prefetches must have a `limit` of at least `limit + offset` of the main query, otherwise you can get an empty result.

## Hybrid Search

One of the most common problems when you have different representations of the same data is to combine the queried points for each representation into a single result.

For example, in text search, it is often useful to combine dense vectors with BM25 full-text search to get the best of both worlds: semantic understanding from dense vectors and precise word matching from BM25.

Qdrant has a few ways of fusing the results from different queries: `rrf` and `dbsf`.

## Reciprocal Rank Fusion (RRF)

RRF considers the positions of results within each query and boosts those that appear closer to the top in multiple sets of results. The score of a document is calculated using its rank in each result set.

The formula uses these parameters:

- D is the set of points across all results.
- R(d) is the set of rankings for a particular document.
- k is a constant (set to 2 by default).
- r is an ordered set of results from one source.
- r_d is the rank of document d in ranking r.
- w_r is the weight of ranking r (set to 1 by default).

Qdrant uses zero-based rank positions; the top result has r_d = 0.

Because w_r defaults to 1, without setting explicit weights, the formula can be simplified to the original RRF function: the score of a document is the sum over its rankings of 1 divided by (k + r_d).

An example of RRF uses two prefetches: a BM25 text query on the `title` payload field and a dense vector query. The main query is `{"rrf": {}}` with a limit of 10.

In the Python client this looks like:

```python
client.query_points(
    collection_name="{collection_name}",
    prefetch=[
        models.Prefetch(query=models.TextQuery(text="time travel"), using="title", limit=20),
        models.Prefetch(query=[0.01, 0.45, 0.67], using="dense", limit=20),
    ],
    query=models.RrfQuery(rrf=models.Rrf()),
    limit=10,
)
```

The key property of RRF is that it fuses by rank, not by raw score. That avoids needing to normalize cosine similarity against BM25 scores, which live on different scales.
