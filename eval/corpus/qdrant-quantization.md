# Quantization (Qdrant)

Quantization is an optional feature in Qdrant that enables efficient storage and search of high-dimensional vectors. By transforming original vectors into new representations, quantization compresses data while preserving close to original relative distances between vectors. Different quantization methods have different mechanics and tradeoffs.

Quantization is primarily used to reduce the memory footprint and accelerate the search process in high-dimensional vector spaces. In the context of Qdrant, quantization allows you to optimize the search engine for specific use cases, striking a balance between accuracy, storage efficiency, and search speed.

There are tradeoffs associated with quantization. On the one hand, quantization allows for significant reductions in storage requirements and faster search times. On the other hand, quantization introduces an approximation error, which can lead to a slight decrease in search quality. The level of this tradeoff depends on the quantization method and its parameters, as well as the characteristics of the data.

Qdrant supports four quantization methods:

- TurboQuant supports 4x to 32x compression, with strong recall across most embedding models.
- Scalar Quantization compresses each vector component from a 32-bit float to an 8-bit integer, achieving 4x compression with minimal accuracy loss.
- Binary Quantization reduces each vector component to one to two bits for up to 32x compression. Best suited for high-dimensional, centered vector distributions.
- Product Quantization enables up to 64x compression when minimizing memory is the top priority.

## How to Choose the Right Quantization Method

Depending on your requirements for recall, compression, and distance metrics:

- Compression 4: use 8-bit TurboQuant. In the benchmarks, the 8-bit TurboQuant encoding matched or exceeded the recall of scalar quantization at the same compression, with similar speed. When using the Manhattan (L1) distance metric, use Scalar Quantization instead.
- Compression 8: use 4-bit TurboQuant. It offers a good balance between recall and compression.
- Compression 16: 2-bit TurboQuant and 2-bit binary quantization offer similar results at this compression level. Binary quantization is faster, but TurboQuant provides better recall.
- Compression ~21: 1.5-bit TurboQuant and 1.5-bit binary quantization offer similar results. Binary quantization is faster, but TurboQuant provides better recall.
- Compression 32: 1-bit TurboQuant and 1-bit binary quantization offer similar results. Binary quantization is faster, but TurboQuant provides better recall.
- Up to 64: use Product Quantization if the memory footprint is the top priority and accuracy and speed are not critical.

## Scalar Quantization

_Available as of v1.1.0_

Scalar quantization compresses vectors by reducing the number of bits used to represent each vector component.

Qdrant uses 32-bit floating numbers to represent the original vector components. Scalar quantization allows you to reduce the number of bits used to 8. In other words, Qdrant performs `float32 -> uint8` conversion for each vector component. Effectively, this means that the amount of memory required to store a vector is reduced by a factor of 4.

In addition to reducing the memory footprint, scalar quantization also speeds up the search process. Qdrant uses a special SIMD CPU instruction to perform fast vector comparison. This instruction works with 8-bit integers, so the conversion to `uint8` allows Qdrant to perform the comparison faster.

The main drawback of scalar quantization is the loss of accuracy. The `float32 -> uint8` conversion introduces an error that can lead to a slight decrease in search quality. However, this error is usually negligible, and tends to be less significant for high-dimensional vectors. In the Qdrant experiments, the error introduced by scalar quantization is usually less than 1%.

## Binary Quantization

_Available as of v1.5.0_

Binary quantization is an extreme case of scalar quantization. This feature lets you represent each vector component as a single bit, effectively reducing the memory footprint by a factor of 32. This is the fastest quantization method, since it lets you perform a vector comparison with a few CPU instructions. Binary quantization can achieve up to a 40x speedup compared to the original vectors.

However, binary quantization is only efficient for high-dimensional vectors and requires a centered distribution of vector components.

At the moment, binary quantization shows good accuracy results with the following models:

- OpenAI `text-embedding-ada-002`, 1536d, tested with the dbpedia dataset, achieving 0.98 recall@100 with 4x oversampling.
- OpenAI `text-embedding-3-large`, 3072d, tested with DBpedia 1M, achieving 0.9966 recall with 3x oversampling and rescoring.
- OpenAI `text-embedding-3-large`, 1536d, tested with DBpedia 1M, achieving 0.9826 recall with 3x oversampling and rescoring.
- OpenAI `text-embedding-3-small`, 1536d, tested with DBpedia 100K, achieving 0.9847 recall with 3x oversampling and rescoring.
- Cohere AI `embed-english-v2.0`, 4096d, tested on Wikipedia embeddings, 0.98 recall@50 with 2x oversampling.

Models with a lower dimensionality or a different distribution of vector components may require additional experiments to find the optimal quantization parameters.

We recommend using binary quantization with rescoring enabled, because the binary pass alone loses accuracy that rescoring recovers. Keep in mind that if the original vectors are stored on disk, rescoring can significantly decrease search speed.

Additionally, oversampling can be used to tune the tradeoff between search speed and search quality at query time.

### Rescoring and Oversampling Results

A 2024 sweep over the OpenAI `text-embedding-3` models shows what rescoring and oversampling do to binary quantization in practice.

Setup: 100,000 vectors sampled at random from the DBpedia OpenAI 1M dataset, embedded with `text-embedding-3-large` at 1024, 1536, and 3072 dimensions and `text-embedding-3-small` at 512, 1024, and 1536 dimensions. 100 randomly selected records served as queries. For each configuration the sweep toggled rescoring, varied the oversampling factor, and ran search limits of 10, 20, 50, and 100. Accuracy is the overlap between the quantized results and exact search, averaged over the queries.

Rescoring improved accuracy in every configuration of this sweep, and the gap was largest at low dimensionality.

- `text-embedding-3-large`, 3072 dimensions: accuracy without rescoring 76% to 77%, with rescoring 97% to 99%.
- `text-embedding-3-small`, 512 dimensions: accuracy without rescoring 53% to 55%, with rescoring 71% to 91%.

Limits of this run: treat the ranges as indicative, not as confidence intervals, because 100 queries is a small sample. The run used 1-bit binary quantization only. Rescoring is not a guarantee of accuracy: it can only reorder candidates the binary pass already retrieved, so it recovers less when the oversampling factor is low or the dimensionality is small, and it reads the original vectors, which costs disk I/O when they are stored on disk.

## 1.5-Bit and 2-Bit Quantization

_Available as of v1.15.0_

Binary quantization storage can use 2 and 1.5 bits per dimension, improving precision for smaller vectors. One-bit compression resulted in significant data loss and precision drops for vectors smaller than a thousand dimensions, often requiring expensive rescoring. 2-bit quantization offers 16X compression compared to 32X with one bit, improving performance for smaller vector dimensions. The 1.5-bit quantization compression offers about 21X compression and intermediate accuracy.

A major limitation of binary quantization is poor handling of values close to zero. 2-bit quantization addresses this by explicitly representing zeros using an efficient scoring mechanism. In the case of 1.5-bit quantization, the zero-bit is shared between two values.

## Asymmetric Quantization

_Available as of v1.15.0_

The Asymmetric Quantization technique allows Qdrant to use different vector encoding algorithms for stored vectors and queries. A particularly interesting combination is binary stored vectors and Scalar quantized queries. With a binary query, every dimension contributes the same ±1 vote to the score, even where the query component is close to zero and its sign is mostly noise. A scalar-quantized query keeps the magnitude of each component, so each dimension's contribution stays close to the exact float32 value.

This approach maintains storage size and RAM usage similar to binary quantization while offering improved precision. It is beneficial for memory-constrained deployments, or where the bottleneck is disk I/O rather than CPU.

## Product Quantization

_Available as of v1.2.0_

Product quantization is a method of compressing vectors to minimize their memory usage by dividing them into chunks and quantizing each segment individually. Each chunk is approximated by a centroid index that represents the original vector component. The positions of the centroids are determined through the utilization of a clustering algorithm such as k-means. For now, Qdrant uses only 256 centroids, so each centroid index can be represented by a single byte.

Product quantization can compress by a more prominent factor than a scalar one. But there are some tradeoffs. Product quantization distance calculations are not SIMD-friendly, so it is slower than scalar quantization. Also, product quantization has a loss of accuracy, so it is recommended to use it only for high-dimensional vectors.

## Setting Up Quantization in Qdrant

You can configure quantization for a collection by specifying the quantization parameters in the `quantization_config` section of the collection configuration.

Quantization will be automatically applied to all vectors during the indexation process. Quantized vectors are stored alongside the original vectors in the collection, so you will still have access to the original vectors if you need them.

The `quantization_config` can also be set on a per vector basis by specifying it in a named vector.

To enable TurboQuant on an existing collection, use a PATCH request or the corresponding `update_collection` method and omit the vector configuration, as it is already defined.
