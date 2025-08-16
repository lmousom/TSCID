# Contributing to TSCID

Thank you for your interest in contributing to TSCID! 

## Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/yourusername/tscid.git`
3. **Create** a branch: `git checkout -b feature-name`
4. **Make** your changes
5. **Test** your changes: `go test -v`
6. **Benchmark** if performance-related: `go run cmd/benchmark/main.go`
7. **Commit** and **push**: `git commit -am "Add feature" && git push origin feature-name`
8. **Create** a Pull Request

## Testing

```bash
# Run all tests
go test -v

# Run benchmarks
go run cmd/benchmark/main.go

# Run examples
cd examples && go run main.go
```

## Guidelines

### Code Style
- Follow standard Go conventions
- Use `gofmt` and `golint`
- Add comments for exported functions
- Keep functions focused and small

### Performance
- Maintain or improve existing performance
- Add benchmarks for new features
- Profile memory allocations for hot paths

### Security
- Preserve cryptographic security properties
- Use secure random number generation
- Avoid timing attacks in comparisons

### Compatibility
- Maintain API compatibility
- Update tests for any changes
- Update documentation and examples

## Bug Reports

Please include:
- Go version
- Operating system
- Minimal reproduction case
- Expected vs actual behavior

## Feature Requests

We welcome feature requests! Please:
- Check existing issues first
- Describe the use case
- Consider backwards compatibility
- Provide implementation ideas if possible

## Documentation

- Update README.md for new features
- Add examples for complex functionality
- Include performance implications
- Update godoc comments

## Pull Request Checklist

- [ ] Tests pass (`go test -v`)
- [ ] Benchmarks pass (`go run cmd/benchmark/main.go`)
- [ ] Examples work (`cd examples && go run main.go`)
- [ ] Documentation updated
- [ ] API compatibility maintained
- [ ] Performance not degraded

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Questions?

Open an issue or start a discussion. We're happy to help!
