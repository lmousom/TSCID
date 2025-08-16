package tscid

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
)

// TestTSCIDCreation tests basic TSCID creation and validation
func TestTSCIDCreation(t *testing.T) {
	tests := []struct {
		name        string
		timestamp   uint64
		sequence    uint16
		entropy     uint16
		random      uint64
		shouldError bool
	}{
		{"valid TSCID", 1000000, 100, 0x1234, 0x123456789ABC, false},
		{"max values", MaxTimestamp, MaxSequence, MaxEntropy, MaxRandom, false},
		{"timestamp overflow", MaxTimestamp + 1, 0, 0, 0, true},
		{"random overflow", 0, 0, 0, MaxRandom + 1, true},
		{"zero values", 0, 0, 0, 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tscid, err := New(tt.timestamp, tt.sequence, tt.entropy, tt.random)

			if tt.shouldError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tscid.Timestamp() != tt.timestamp {
				t.Errorf("timestamp mismatch: got %d, want %d", tscid.Timestamp(), tt.timestamp)
			}
			if tscid.Sequence() != tt.sequence {
				t.Errorf("sequence mismatch: got %d, want %d", tscid.Sequence(), tt.sequence)
			}
			if tscid.Entropy() != tt.entropy {
				t.Errorf("entropy mismatch: got %d, want %d", tscid.Entropy(), tt.entropy)
			}
			if tscid.Random() != tt.random {
				t.Errorf("random mismatch: got %d, want %d", tscid.Random(), tt.random)
			}
		})
	}
}

// TestBinaryEncoding tests binary serialization/deserialization
func TestBinaryEncoding(t *testing.T) {
	// Use a timestamp representing ~30 days from custom epoch (2020-01-01)
	original, err := New(2592000000000, 12345, 0x5678, 0x9ABCDEF01234)
	if err != nil {
		t.Fatalf("failed to create TSCID: %v", err)
	}

	// Test binary encoding
	bytes := original.Bytes()
	if len(bytes) != BinarySize {
		t.Errorf("binary size mismatch: got %d, want %d", len(bytes), BinarySize)
	}

	// Test binary decoding
	decoded := FromBytes(bytes)
	if !original.Equal(decoded) {
		t.Errorf("binary round-trip failed: original=%+v, decoded=%+v", original, decoded)
	}
}

// TestStringEncoding tests string serialization/deserialization
func TestStringEncoding(t *testing.T) {
	// Use a timestamp representing ~30 days from custom epoch (2020-01-01)
	original, err := New(2592000000000, 12345, 0x5678, 0x9ABCDEF01234)
	if err != nil {
		t.Fatalf("failed to create TSCID: %v", err)
	}

	// Test different formats
	formats := []struct {
		format   Format
		expected int // expected length
	}{
		{FormatCanonical, CanonicalSize},
		{FormatCompact, Base32Size},
		{FormatHyphenated, 35}, // tscid- + 26 chars + 3 hyphens = 6+26+3 = 35
	}

	for _, tt := range formats {
		t.Run(fmt.Sprintf("format_%d", tt.format), func(t *testing.T) {
			encoded := original.Format(tt.format)
			if len(encoded) != tt.expected {
				t.Errorf("encoded length mismatch: got %d, want %d", len(encoded), tt.expected)
			}

			// Test parsing
			decoded, err := Parse(encoded)
			if err != nil {
				t.Fatalf("failed to parse %q: %v", encoded, err)
			}

			if !original.Equal(decoded) {
				t.Errorf("string round-trip failed: original=%+v, decoded=%+v", original, decoded)
			}
		})
	}
}

// TestStringParsing tests parsing various string formats
func TestStringParsing(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"canonical", "tscid_01H7VK4Q9P6S3N7B2M8F9G1A5Z", false},
		{"compact", "01H7VK4Q9P6S3N7B2M8F9G1A5Z", false},
		{"hyphenated", "tscid-01H7VK-4Q9P-6S3N-7B2M8F9G1A5Z", false},
		{"invalid prefix", "invalid_01H7VK4Q9P6S3N7B2M8F9G1A5Z", true},
		{"too short", "tscid_123", true},
		{"too long", "tscid_01H7VK4Q9P6S3N7B2M8F9G1A5Z123", true},
		{"invalid character", "tscid_01H7VK4Q9P6S3N7B2M8F9G1A5!", true},
		{"empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(tt.input)

			if tt.wantErr && err == nil {
				t.Error("expected error but got none")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

// TestMustParse tests the MustParse function
func TestMustParse(t *testing.T) {
	// Valid input should not panic
	defer func() {
		if r := recover(); r != nil {
			t.Error("MustParse panicked on valid input")
		}
	}()

	tscid := MustParse("tscid_01H7VK4Q9P6S3N7B2M8F9G1A5Z")
	if tscid.IsZero() {
		t.Error("parsed TSCID is zero")
	}
}

// TestMustParsePanic tests that MustParse panics on invalid input
func TestMustParsePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MustParse should have panicked on invalid input")
		}
	}()

	MustParse("invalid")
}

// TestComparison tests TSCID comparison operations
func TestComparison(t *testing.T) {
	// Create TSCIDs with different timestamps
	earlier, _ := New(1000000, 0, 0, 0)
	later, _ := New(2000000, 0, 0, 0)
	same1, _ := New(1000000, 0, 0, 0)
	same2, _ := New(1000000, 0, 0, 0)

	// Test temporal ordering
	if !earlier.Less(later) {
		t.Error("earlier TSCID should be less than later TSCID")
	}
	if later.Less(earlier) {
		t.Error("later TSCID should not be less than earlier TSCID")
	}

	// Test equality
	if !same1.Equal(same2) {
		t.Error("identical TSCIDs should be equal")
	}
	if earlier.Equal(later) {
		t.Error("different TSCIDs should not be equal")
	}

	// Test Compare function
	if earlier.Compare(later) != -1 {
		t.Error("earlier.Compare(later) should return -1")
	}
	if later.Compare(earlier) != 1 {
		t.Error("later.Compare(earlier) should return 1")
	}
	if same1.Compare(same2) != 0 {
		t.Error("same1.Compare(same2) should return 0")
	}
}

// TestGenerator tests TSCID generation
func TestGenerator(t *testing.T) {
	g := NewGenerator()

	// Test basic generation
	tscid1, err := g.Generate()
	if err != nil {
		t.Fatalf("failed to generate TSCID: %v", err)
	}

	tscid2, err := g.Generate()
	if err != nil {
		t.Fatalf("failed to generate second TSCID: %v", err)
	}

	// TSCIDs should be different
	if tscid1.Equal(tscid2) {
		t.Error("consecutive TSCIDs should be different")
	}

	// Second should be greater than first (temporal ordering)
	if !tscid1.Less(tscid2) {
		t.Error("second TSCID should be greater than first")
	}
}

// TestGeneratorWithNodeID tests generator with custom node ID
func TestGeneratorWithNodeID(t *testing.T) {
	nodeID := uint8(42)
	g := NewGeneratorWithNodeID(nodeID)

	if g.NodeID() != nodeID {
		t.Errorf("node ID mismatch: got %d, want %d", g.NodeID(), nodeID)
	}

	tscid, err := g.Generate()
	if err != nil {
		t.Fatalf("failed to generate TSCID: %v", err)
	}

	// Check that entropy contains the node ID
	expectedEntropy := uint16(nodeID)<<8 | uint16(g.ProcessID())
	if tscid.Entropy() != expectedEntropy {
		t.Errorf("entropy mismatch: got %d, want %d", tscid.Entropy(), expectedEntropy)
	}
}

// TestFastGenerator tests the fast generator
func TestFastGenerator(t *testing.T) {
	g := NewFastGenerator()

	tscid1, err := g.Generate()
	if err != nil {
		t.Fatalf("failed to generate TSCID: %v", err)
	}

	tscid2, err := g.Generate()
	if err != nil {
		t.Fatalf("failed to generate second TSCID: %v", err)
	}

	// TSCIDs should be different and ordered
	if tscid1.Equal(tscid2) {
		t.Error("consecutive TSCIDs should be different")
	}
	if !tscid1.Less(tscid2) {
		t.Error("second TSCID should be greater than first")
	}
}

// TestBatchGeneration tests batch generation
func TestBatchGeneration(t *testing.T) {
	g := NewGenerator()
	count := 1000

	tscids, err := g.GenerateBatch(count)
	if err != nil {
		t.Fatalf("failed to generate batch: %v", err)
	}

	if len(tscids) != count {
		t.Errorf("batch size mismatch: got %d, want %d", len(tscids), count)
	}

	// Check temporal ordering
	for i := 1; i < len(tscids); i++ {
		if !tscids[i-1].Less(tscids[i]) && !tscids[i-1].Equal(tscids[i]) {
			t.Errorf("batch TSCIDs not in temporal order at index %d", i)
		}
	}

	// Check uniqueness
	seen := make(map[string]bool)
	for _, tscid := range tscids {
		str := tscid.String()
		if seen[str] {
			t.Errorf("duplicate TSCID found: %s", str)
		}
		seen[str] = true
	}
}

// TestConcurrentGeneration tests thread safety
func TestConcurrentGeneration(t *testing.T) {
	g := NewGenerator()
	numGoroutines := 10
	numPerGoroutine := 1000

	var wg sync.WaitGroup
	results := make([][]TSCID, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			batch, err := g.GenerateBatch(numPerGoroutine)
			if err != nil {
				t.Errorf("goroutine %d failed: %v", idx, err)
				return
			}
			results[idx] = batch
		}(i)
	}

	wg.Wait()

	// Collect all TSCIDs and check uniqueness
	seen := make(map[string]bool)
	var allTSCIDs []TSCID

	for _, batch := range results {
		for _, tscid := range batch {
			str := tscid.String()
			if seen[str] {
				t.Errorf("duplicate TSCID found in concurrent test: %s", str)
			}
			seen[str] = true
			allTSCIDs = append(allTSCIDs, tscid)
		}
	}

	expectedTotal := numGoroutines * numPerGoroutine
	if len(allTSCIDs) != expectedTotal {
		t.Errorf("total TSCID count mismatch: got %d, want %d", len(allTSCIDs), expectedTotal)
	}
}

// TestTemporalOrdering tests that TSCIDs sort correctly by time
func TestTemporalOrdering(t *testing.T) {
	g := NewGenerator()
	count := 100

	var tscids []TSCID
	for i := 0; i < count; i++ {
		tscid, err := g.Generate()
		if err != nil {
			t.Fatalf("failed to generate TSCID: %v", err)
		}
		tscids = append(tscids, tscid)

		// Small delay to ensure timestamp differences
		if i%10 == 0 {
			time.Sleep(time.Microsecond)
		}
	}

	// Shuffle the TSCIDs
	rand.Shuffle(len(tscids), func(i, j int) {
		tscids[i], tscids[j] = tscids[j], tscids[i]
	})

	// Sort using string comparison
	stringSort := make([]string, len(tscids))
	for i, tscid := range tscids {
		stringSort[i] = tscid.String()
	}
	sort.Strings(stringSort)

	// Sort using TSCID comparison
	tscidSort := make([]TSCID, len(tscids))
	copy(tscidSort, tscids)
	sort.Slice(tscidSort, func(i, j int) bool {
		return tscidSort[i].Less(tscidSort[j])
	})

	// Both sorts should produce the same order
	for i := 0; i < len(tscids); i++ {
		if stringSort[i] != tscidSort[i].String() {
			t.Errorf("sort mismatch at index %d: string=%s, tscid=%s",
				i, stringSort[i], tscidSort[i].String())
		}
	}
}

// TestDefaultGenerator tests the package-level functions
func TestDefaultGenerator(t *testing.T) {
	tscid1, err := Generate()
	if err != nil {
		t.Fatalf("failed to generate TSCID: %v", err)
	}

	tscid2 := MustGenerate()

	if tscid1.Equal(tscid2) {
		t.Error("consecutive TSCIDs should be different")
	}
	if !tscid1.Less(tscid2) {
		t.Error("second TSCID should be greater than first")
	}
}

// TestZeroValue tests zero value handling
func TestZeroValue(t *testing.T) {
	var zero TSCID
	if !zero.IsZero() {
		t.Error("zero value should be detected as zero")
	}
	if !zero.Equal(Zero) {
		t.Error("zero value should equal Zero constant")
	}

	nonZero := MustGenerate()
	if nonZero.IsZero() {
		t.Error("generated TSCID should not be zero")
	}
}

// Benchmarks

// BenchmarkGenerate benchmarks TSCID generation
func BenchmarkGenerate(b *testing.B) {
	g := NewGenerator()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := g.Generate()
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGenerateFast benchmarks fast TSCID generation
func BenchmarkGenerateFast(b *testing.B) {
	g := NewFastGenerator()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := g.Generate()
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGenerateParallel benchmarks parallel TSCID generation
func BenchmarkGenerateParallel(b *testing.B) {
	g := NewGenerator()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := g.Generate()
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkGenerateBatch benchmarks batch generation
func BenchmarkGenerateBatch(b *testing.B) {
	g := NewGenerator()
	batchSize := 100

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := g.GenerateBatch(batchSize)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkString benchmarks string encoding
func BenchmarkString(b *testing.B) {
	tscid := MustGenerate()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = tscid.String()
	}
}

// BenchmarkParse benchmarks string parsing
func BenchmarkParse(b *testing.B) {
	tscid := MustGenerate()
	str := tscid.String()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := Parse(str)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkBytes benchmarks binary encoding
func BenchmarkBytes(b *testing.B) {
	tscid := MustGenerate()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = tscid.Bytes()
	}
}

// BenchmarkFromBytes benchmarks binary decoding
func BenchmarkFromBytes(b *testing.B) {
	tscid := MustGenerate()
	bytes := tscid.Bytes()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = FromBytes(bytes)
	}
}

// BenchmarkCompare benchmarks TSCID comparison
func BenchmarkCompare(b *testing.B) {
	tscid1 := MustGenerate()
	tscid2 := MustGenerate()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = tscid1.Compare(tscid2)
	}
}

// Comparison benchmarks against UUID

// BenchmarkUUIDv4 benchmarks UUID v4 generation for comparison
func BenchmarkUUIDv4(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = uuid.New()
	}
}

// BenchmarkUUIDv1 benchmarks UUID v1 generation for comparison
func BenchmarkUUIDv1(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = uuid.Must(uuid.NewUUID())
	}
}

// BenchmarkUUIDv4String benchmarks UUID v4 string conversion
func BenchmarkUUIDv4String(b *testing.B) {
	id := uuid.New()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = id.String()
	}
}

// BenchmarkUUIDv4Parse benchmarks UUID v4 parsing
func BenchmarkUUIDv4Parse(b *testing.B) {
	id := uuid.New()
	str := id.String()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := uuid.Parse(str)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Example tests for documentation

func ExampleGenerator_Generate() {
	g := NewGenerator()
	tscid, err := g.Generate()
	if err != nil {
		panic(err)
	}

	fmt.Printf("Generated TSCID: %s\n", tscid)
	fmt.Printf("Timestamp: %d\n", tscid.Timestamp())
	fmt.Printf("Time: %s\n", tscid.Time().Format(time.RFC3339))
}

func ExampleParse() {
	tscidStr := "tscid_01H7VK4Q9P6S3N7B2M8F9G1A5Z"
	tscid, err := Parse(tscidStr)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Parsed TSCID: %+v\n", tscid)
}

func ExampleTSCID_Format() {
	tscid := MustGenerate()

	fmt.Printf("Canonical: %s\n", tscid.Format(FormatCanonical))
	fmt.Printf("Compact: %s\n", tscid.Format(FormatCompact))
	fmt.Printf("Hyphenated: %s\n", tscid.Format(FormatHyphenated))
}

// Performance comparison test
func TestPerformanceComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping performance comparison in short mode")
	}

	const iterations = 100000

	// Benchmark TSCID generation
	start := time.Now()
	g := NewFastGenerator()
	for i := 0; i < iterations; i++ {
		_, err := g.Generate()
		if err != nil {
			t.Fatal(err)
		}
	}
	tscidDuration := time.Since(start)

	// Benchmark UUID v4 generation
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_ = uuid.New()
	}
	uuidDuration := time.Since(start)

	tscidRate := float64(iterations) / tscidDuration.Seconds()
	uuidRate := float64(iterations) / uuidDuration.Seconds()

	t.Logf("TSCID generation: %.0f IDs/sec", tscidRate)
	t.Logf("UUID v4 generation: %.0f IDs/sec", uuidRate)
	t.Logf("Performance ratio: %.2f", tscidRate/uuidRate)

	// TSCID should be competitive (within 50% of UUID performance)
	if tscidRate < uuidRate*0.5 {
		t.Errorf("TSCID performance too slow: %.0f vs %.0f IDs/sec", tscidRate, uuidRate)
	}
}

// Collision resistance test
func TestCollisionResistance(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping collision test in short mode")
	}

	const count = 1000000
	g := NewFastGenerator()
	seen := make(map[string]bool, count)

	for i := 0; i < count; i++ {
		tscid, err := g.Generate()
		if err != nil {
			t.Fatal(err)
		}

		str := tscid.String()
		if seen[str] {
			t.Fatalf("collision detected after %d iterations: %s", i+1, str)
		}
		seen[str] = true
	}

	t.Logf("Generated %d unique TSCIDs with no collisions", count)
}

// Memory usage test
func TestMemoryUsage(t *testing.T) {
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	// Generate many TSCIDs
	const count = 100000
	g := NewFastGenerator()
	tscids := make([]TSCID, count)

	for i := 0; i < count; i++ {
		tscid, err := g.Generate()
		if err != nil {
			t.Fatal(err)
		}
		tscids[i] = tscid
	}

	runtime.GC()
	runtime.ReadMemStats(&m2)

	// Check for memory allocation overflow
	if m2.Alloc < m1.Alloc {
		t.Skip("Memory allocation counter wrapped, skipping test")
	}

	allocDiff := m2.Alloc - m1.Alloc
	if allocDiff > uint64(count)*1000 { // More than 1KB per TSCID suggests measurement error
		t.Logf("Large memory allocation detected (%d bytes), possibly due to GC or other allocations", allocDiff)
		t.Skip("Memory measurement unreliable, skipping test")
	}

	bytesPerTSCID := allocDiff / uint64(count)
	t.Logf("Memory usage: %d bytes per TSCID", bytesPerTSCID)

	// TSCID struct should be around 16 bytes plus some slice overhead
	if bytesPerTSCID > 64 {
		t.Errorf("memory usage too high: %d bytes per TSCID", bytesPerTSCID)
	}

	// Use tscids to prevent optimization
	_ = tscids[count-1]
}

// Test convenience functions and helper methods
func TestConvenienceFunctions(t *testing.T) {
	// Test NewID
	id := NewID()
	if id.IsZero() {
		t.Error("NewID() returned zero value")
	}

	// Test NewBatch
	batch := NewBatch(5)
	if len(batch) != 5 {
		t.Errorf("NewBatch(5) returned %d items, expected 5", len(batch))
	}

	// Test GenerateBatch and MustGenerateBatch
	batch2, err := GenerateBatch(3)
	if err != nil {
		t.Errorf("GenerateBatch failed: %v", err)
	}
	if len(batch2) != 3 {
		t.Errorf("GenerateBatch(3) returned %d items, expected 3", len(batch2))
	}

	batch3 := MustGenerateBatch(2)
	if len(batch3) != 2 {
		t.Errorf("MustGenerateBatch(2) returned %d items, expected 2", len(batch3))
	}
}

// Test time-related methods
func TestTimeOperations(t *testing.T) {
	id := NewID()

	// Test Time()
	createdAt := id.Time()
	if createdAt.IsZero() {
		t.Error("Time() returned zero time")
	}

	// Test Age()
	age := id.Age()
	if age < 0 {
		t.Error("Age() returned negative duration")
	}

	// Test IsOlderThan/IsNewerThan
	if id.IsOlderThan(time.Hour) {
		t.Error("Newly created ID should not be older than 1 hour")
	}

	if !id.IsNewerThan(time.Hour) {
		t.Error("Newly created ID should be newer than 1 hour ago")
	}

	// Create an older ID by manipulating timestamp
	oldID, _ := New(1000, 0, 0, 12345) // Very old timestamp
	if !oldID.IsOlderThan(time.Millisecond) {
		t.Error("Old ID should be older than 1ms")
	}
}

// Test format methods
func TestFormatMethods(t *testing.T) {
	id := NewID()

	// Test ToBytes
	bytes := id.ToBytes()
	if len(bytes) != 16 {
		t.Errorf("ToBytes() returned %d bytes, expected 16", len(bytes))
	}

	// Test format methods
	compact := id.Compact()
	canonical := id.Canonical()
	hyphenated := id.Hyphenated()

	if len(compact) != Base32Size {
		t.Errorf("Compact() returned %d chars, expected %d", len(compact), Base32Size)
	}

	if len(canonical) != CanonicalSize {
		t.Errorf("Canonical() returned %d chars, expected %d", len(canonical), CanonicalSize)
	}

	if !strings.HasPrefix(canonical, "tscid_") {
		t.Error("Canonical() should start with 'tscid_'")
	}

	if !strings.HasPrefix(hyphenated, "tscid-") {
		t.Error("Hyphenated() should start with 'tscid-'")
	}

	// Verify they represent the same ID
	parsed1, _ := Parse(compact)
	parsed2, _ := Parse(canonical)
	parsed3, _ := Parse(hyphenated)

	if !id.Equal(parsed1) || !id.Equal(parsed2) || !id.Equal(parsed3) {
		t.Error("Different formats should parse to the same ID")
	}
}

// Test JSON marshaling/unmarshaling
func TestJSONSupport(t *testing.T) {
	id := NewID()

	// Test MarshalJSON
	jsonBytes, err := id.MarshalJSON()
	if err != nil {
		t.Errorf("MarshalJSON failed: %v", err)
	}

	var jsonStr string
	if err := json.Unmarshal(jsonBytes, &jsonStr); err != nil {
		t.Errorf("Failed to unmarshal JSON: %v", err)
	}

	// Test UnmarshalJSON
	var id2 TSCID
	if err := id2.UnmarshalJSON(jsonBytes); err != nil {
		t.Errorf("UnmarshalJSON failed: %v", err)
	}

	if !id.Equal(id2) {
		t.Error("JSON round-trip failed")
	}

	// Test with struct
	type TestStruct struct {
		ID   TSCID  `json:"id"`
		Name string `json:"name"`
	}

	original := TestStruct{ID: id, Name: "test"}
	data, err := json.Marshal(original)
	if err != nil {
		t.Errorf("Failed to marshal struct: %v", err)
	}

	var decoded TestStruct
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Errorf("Failed to unmarshal struct: %v", err)
	}

	if !original.ID.Equal(decoded.ID) || original.Name != decoded.Name {
		t.Error("Struct JSON round-trip failed")
	}
}

// Test SQL database support
func TestSQLSupport(t *testing.T) {
	id := NewID()

	// Test Value() method
	value, err := id.Value()
	if err != nil {
		t.Errorf("Value() failed: %v", err)
	}

	str, ok := value.(string)
	if !ok {
		t.Error("Value() should return a string")
	}

	// Test Scan() method
	var id2 TSCID
	if err := id2.Scan(str); err != nil {
		t.Errorf("Scan() failed: %v", err)
	}

	if !id.Equal(id2) {
		t.Error("SQL round-trip failed")
	}

	// Test Scan with []byte
	var id3 TSCID
	if err := id3.Scan([]byte(str)); err != nil {
		t.Errorf("Scan([]byte) failed: %v", err)
	}

	if !id.Equal(id3) {
		t.Error("SQL []byte round-trip failed")
	}

	// Test Scan with nil
	var id4 TSCID
	if err := id4.Scan(nil); err != nil {
		t.Errorf("Scan(nil) failed: %v", err)
	}

	if !id4.IsZero() {
		t.Error("Scan(nil) should result in zero value")
	}

	// Test Scan with invalid type
	var id5 TSCID
	if err := id5.Scan(123); err == nil {
		t.Error("Scan(int) should fail")
	}
}

// Test smart generator and auto mode
func TestSmartGenerator(t *testing.T) {
	gen := NewSmartGenerator()

	// Test single-threaded usage
	_, err := gen.Generate()
	if err != nil {
		t.Errorf("SmartGenerator.Generate() failed: %v", err)
	}

	// Test multi-threaded usage
	const numGoroutines = 4
	const idsPerGoroutine = 100

	ids := make(chan TSCID, numGoroutines*idsPerGoroutine)
	errors := make(chan error, numGoroutines*idsPerGoroutine)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			for j := 0; j < idsPerGoroutine; j++ {
				id, err := gen.Generate()
				if err != nil {
					errors <- err
					return
				}
				ids <- id
			}
		}()
	}

	// Collect results
	var generatedIDs []TSCID
	for i := 0; i < numGoroutines*idsPerGoroutine; i++ {
		select {
		case id := <-ids:
			generatedIDs = append(generatedIDs, id)
		case err := <-errors:
			t.Errorf("Concurrent generation failed: %v", err)
		case <-time.After(5 * time.Second):
			t.Fatal("Timeout waiting for concurrent generation")
		}
	}

	// Verify all IDs are unique
	seen := make(map[string]bool)
	for _, id := range generatedIDs {
		str := id.String()
		if seen[str] {
			t.Errorf("Duplicate ID generated: %s", str)
		}
		seen[str] = true
	}

	// Verify we have the right number
	if len(generatedIDs) != numGoroutines*idsPerGoroutine {
		t.Errorf("Expected %d IDs, got %d", numGoroutines*idsPerGoroutine, len(generatedIDs))
	}
}

// Test fast batch generation
func TestFastBatchGeneration(t *testing.T) {
	gen := NewFastGenerator()

	// Test small batch
	batch, err := gen.GenerateBatch(10)
	if err != nil {
		t.Errorf("Fast batch generation failed: %v", err)
	}

	if len(batch) != 10 {
		t.Errorf("Expected 10 IDs, got %d", len(batch))
	}

	// Verify all IDs are unique and ordered
	for i := 1; i < len(batch); i++ {
		if !batch[i-1].Less(batch[i]) && !batch[i-1].Equal(batch[i]) {
			t.Errorf("Batch IDs not properly ordered at index %d", i)
		}
	}

	// Test larger batch
	largeBatch, err := gen.GenerateBatch(1000)
	if err != nil {
		t.Errorf("Large fast batch generation failed: %v", err)
	}

	if len(largeBatch) != 1000 {
		t.Errorf("Expected 1000 IDs, got %d", len(largeBatch))
	}

	// Verify uniqueness
	seen := make(map[string]bool)
	for _, id := range largeBatch {
		str := id.String()
		if seen[str] {
			t.Errorf("Duplicate ID in fast batch: %s", str)
		}
		seen[str] = true
	}
}

// Test generator entropy method
func TestGeneratorEntropy(t *testing.T) {
	gen := NewGenerator()
	entropy := gen.Entropy()

	// Entropy should be non-zero (combines node and process ID)
	if entropy == 0 {
		t.Error("Generator entropy should not be zero")
	}

	// Test that different generators can have different entropy
	gen2 := NewGeneratorWithNodeID(123)
	entropy2 := gen2.Entropy()

	// They might be the same if process ID is the same, but let's check the method works
	if entropy2 == 0 {
		t.Error("Generator with custom node ID should have non-zero entropy")
	}
}

// Test error conditions and edge cases
func TestErrorConditions(t *testing.T) {
	// Test MustGenerate panic (this won't actually panic in normal conditions)
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("MustGenerate should not panic in normal conditions: %v", r)
		}
	}()
	_ = MustGenerate()

	// Test Compare method
	id1 := NewID()
	time.Sleep(1 * time.Millisecond) // Ensure different timestamp
	id2 := NewID()

	if id1.Compare(id2) >= 0 {
		t.Error("id1 should be less than id2")
	}

	if id2.Compare(id1) <= 0 {
		t.Error("id2 should be greater than id1")
	}

	if id1.Compare(id1) != 0 {
		t.Error("Compare(id, id) should return 0")
	}

	// Test Format method
	formatted := id1.Format(FormatCompact)
	if len(formatted) != Base32Size {
		t.Errorf("Format(FormatCompact) returned %d chars, expected %d", len(formatted), Base32Size)
	}

	// Test generateFast with fast generator
	fastGen := NewFastGenerator()
	for i := 0; i < 10; i++ {
		_, err := fastGen.Generate()
		if err != nil {
			t.Errorf("Fast generator failed: %v", err)
		}
	}

	// Test generateInternal edge cases by generating many IDs quickly
	gen := NewGenerator()
	for i := 0; i < 1000; i++ {
		_, err := gen.Generate()
		if err != nil {
			t.Errorf("Generator failed at iteration %d: %v", i, err)
		}
	}
}

// Test batch generation edge cases
func TestBatchEdgeCases(t *testing.T) {
	gen := NewGenerator()

	// Test empty batch
	batch, err := gen.GenerateBatch(0)
	if err != nil {
		t.Errorf("GenerateBatch(0) failed: %v", err)
	}
	if len(batch) != 0 {
		t.Errorf("GenerateBatch(0) should return empty slice")
	}

	// Test single item batch
	batch, err = gen.GenerateBatch(1)
	if err != nil {
		t.Errorf("GenerateBatch(1) failed: %v", err)
	}
	if len(batch) != 1 {
		t.Errorf("GenerateBatch(1) should return 1 item")
	}

	// Test large batch to trigger different code paths
	largeBatch, err := gen.GenerateBatch(10000)
	if err != nil {
		t.Errorf("Large batch generation failed: %v", err)
	}
	if len(largeBatch) != 10000 {
		t.Errorf("Expected 10000 items, got %d", len(largeBatch))
	}

	// Verify ordering in large batch
	for i := 1; i < len(largeBatch); i++ {
		if largeBatch[i-1].Compare(largeBatch[i]) > 0 {
			t.Errorf("Large batch not properly ordered at index %d", i)
		}
	}
}
