# TSCID - Time-Sorted Collision-resistant IDs

[![Go Reference](https://pkg.go.dev/badge/github.com/lmousom/tscid.svg)](https://pkg.go.dev/github.com/lmousom/tscid)
[![Go Report Card](https://goreportcard.com/badge/github.com/lmousom/tscid)](https://goreportcard.com/report/github.com/lmousom/tscid)
[![codecov](https://codecov.io/gh/lmousom/tscid/branch/main/graph/badge.svg)](https://codecov.io/gh/lmousom/tscid)

TSCIDs are 128-bit identifiers that sort by creation time. They're faster to generate than UUIDs and work better with databases because they're naturally ordered.

Each TSCID contains a microsecond timestamp, sequence counter, node identifier, and random data. This gives you collision resistance while maintaining temporal ordering - something regular UUIDs can't do.

Since TSCIDs are ordered, database indexes stay efficient and range queries work intuitively with time-based data.

## Quick Start

```bash
go get github.com/lmousom/tscid
```

```go
package main

import (
    "fmt"
    "github.com/lmousom/tscid"
)

func main() {
    // Generate a single ID
    id := tscid.NewID()
    fmt.Println(id) // tscid_51DE123ABC0003C8XYZ789...
    
    // Generate multiple IDs efficiently
    batch := tscid.NewBatch(100)
    fmt.Printf("Generated %d IDs\n", len(batch))
}
```

## API Reference

### Core Functions

```go
// Simple ID generation
id := tscid.NewID()                    // Most common usage
ids := tscid.NewBatch(1000)           // Efficient batch generation

// With error handling
id, err := tscid.Generate()           // Returns error if any
id := tscid.MustGenerate()            // Panics on error

// Parsing
id, err := tscid.Parse("tscid_...")   // Parse from string
id := tscid.MustParse("tscid_...")    // Parse or panic
```

### TSCID Methods

```go
id := tscid.NewID()

// String formats
id.String()                           // tscid_ABC123... (default)
id.Compact()                          // ABC123... (no prefix)
id.Canonical()                        // tscid_ABC123... (explicit)
id.Hyphenated()                       // tscid-ABC-123-456-789...

// Time operations
id.Time()                             // time.Time when created
id.Age()                              // time.Duration since creation
id.IsOlderThan(time.Hour)             // true/false
id.IsNewerThan(time.Minute)           // true/false

// Comparison
id1.Compare(id2)                      // -1, 0, or 1
id1.Less(id2)                         // true/false
id1.Equal(id2)                        // true/false

// Binary representation
bytes := id.Bytes()                   // [16]byte
bytes := id.ToBytes()                 // []byte
```

## Integration

### JSON Support

```go
type User struct {
    ID   tscid.TSCID `json:"id"`
    Name string      `json:"name"`
}

user := User{ID: tscid.NewID(), Name: "Alice"}
data, _ := json.Marshal(user)         // {"id":"tscid_...","name":"Alice"}
```

### Database Support

```go
type Product struct {
    ID    tscid.TSCID `db:"id"`
    Name  string      `db:"name"`
    Price float64     `db:"price"`
}

// Works with any SQL driver
_, err := db.Exec("INSERT INTO products (id, name, price) VALUES (?, ?, ?)",
    tscid.NewID(), "Widget", 29.99)
```

### HTTP/REST APIs

```go
func createUser(w http.ResponseWriter, r *http.Request) {
    user := User{
        ID:   tscid.NewID(),
        Name: r.FormValue("name"),
    }
    
    // Store in database...
    
    w.Header().Set("Location", "/users/"+user.ID.String())
    json.NewEncoder(w).Encode(user)
}
```

## Performance

Based on benchmarks on Apple M2:

| Operation | TSCID | UUID v4 | Improvement |
|-----------|-------|---------|-------------|
| Single generation | 8.4M/sec | 3.9M/sec | 2.1x faster |
| Fast generator | 14.3M/sec | 3.9M/sec | 3.6x faster |
| Batch (10K) | 16M/sec | 4M/sec | 4x faster |
| Sorting | ~5ms | ~5ms | Similar |

Results vary by system. Run `go run cmd/benchmark/main.go` to test on your hardware.

## Advanced Usage

### Custom Generators

```go
// Thread-safe generator (default)
gen := tscid.NewGenerator()
id, err := gen.Generate()

// Single-threaded optimized
fastGen := tscid.NewFastGenerator()
id, err := fastGen.Generate()

// Batch generation
ids, err := gen.GenerateBatch(10000)
```

### Error Handling

```go
// With error handling
id, err := tscid.Generate()
if err != nil {
    log.Fatal(err)
}

// Panic on error (most common)
id := tscid.NewID()
```

## Architecture

TSCID structure (128 bits):
```
┌─────────────┬──────────┬─────────┬──────────────┐
│ Timestamp   │ Sequence │ Entropy │ Random       │
│ (48 bits)   │ (16 bits)│ (16 bits)│ (48 bits)   │
└─────────────┴──────────┴─────────┴──────────────┘
```

- **Timestamp**: Microseconds since 2020-01-01 (good until year ~10,920)
- **Sequence**: Counter for IDs created in the same microsecond
- **Entropy**: Node + process ID to avoid collisions across machines
- **Random**: ChaCha8 random data for collision resistance

## Security

TSCIDs use ChaCha8 for random number generation, which is cryptographically secure and fast. Each generator is seeded from the system's random source.

While the timestamp portion is predictable (that's the point), the random portion ensures you can't guess future IDs. The combination of timestamp, sequence, entropy, and random data makes collisions extremely unlikely.

## Comparison

| Feature | TSCID | UUID v4 | UUID v1 | ULID |
|---------|-------|---------|---------|------|
| Temporal Ordering | Yes | No | Partial | Yes |
| Collision Resistant | Yes | Yes | Partial | Yes |
| Database Optimized | Yes | No | No | Yes |
| Cryptographically Secure | Yes | Yes | No | Yes |
| High Performance | Yes | No | Yes | Partial |
| Standard Format | No | Yes | Yes | No |

## Contributing

Pull requests and issues welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [Documentation](https://pkg.go.dev/github.com/lmousom/tscid)
- [Examples](examples/)
- [Benchmarks](cmd/benchmark/)
- [Issues](https://github.com/lmousom/tscid/issues)
