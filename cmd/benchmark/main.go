package main

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"sort"
	"time"

	"github.com/google/uuid"
	"github.com/lmousom/tscid"
)

func main() {
	fmt.Println("TSCID Benchmark")
	fmt.Println("===============")
	fmt.Printf("Go %s on %s/%s, %d cores\n",
		runtime.Version(), runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	fmt.Println()

	runGenerationBenchmark()
	runBatchBenchmark()
	runConcurrentBenchmark()
	runOrderingTest()
	runStringBenchmark()
	runCollisionTest()
	runMemoryTest()
	runSortingComparison()
}

func runGenerationBenchmark() {
	fmt.Println("Generation Performance")
	fmt.Println("---------------------")

	const iterations = 500000

	// Create generators once, reuse them
	regularGen := tscid.NewGenerator()
	fastGen := tscid.NewFastGenerator()

	tests := []struct {
		name string
		fn   func() error
	}{
		{"TSCID", func() error {
			_, err := regularGen.Generate()
			return err
		}},
		{"TSCID Fast", func() error {
			_, err := fastGen.Generate()
			return err
		}},
		{"UUID v4", func() error {
			_ = uuid.New()
			return nil
		}},
	}

	for _, test := range tests {
		// Warmup
		for i := 0; i < 1000; i++ {
			test.fn()
		}

		start := time.Now()
		for i := 0; i < iterations; i++ {
			if err := test.fn(); err != nil {
				log.Printf("Error in %s: %v", test.name, err)
			}
		}
		elapsed := time.Since(start)

		opsPerSec := float64(iterations) / elapsed.Seconds()
		nsPerOp := elapsed.Nanoseconds() / int64(iterations)

		fmt.Printf("%-12s %8.0f ops/sec  %6d ns/op\n", test.name, opsPerSec, nsPerOp)
	}
	fmt.Println()
}

func runBatchBenchmark() {
	fmt.Println("Batch Generation")
	fmt.Println("----------------")

	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		fmt.Printf("Batch size %d:\n", size)

		g := tscid.NewGenerator()
		start := time.Now()
		_, err := g.GenerateBatch(size)
		if err != nil {
			log.Printf("Batch error: %v", err)
			continue
		}
		elapsed := time.Since(start)

		rate := float64(size) / elapsed.Seconds()
		fmt.Printf("  TSCID: %8.0f IDs/sec\n", rate)

		// UUID doesn't have batch generation, so single generation
		start = time.Now()
		for i := 0; i < size; i++ {
			_ = uuid.New()
		}
		elapsed = time.Since(start)
		rate = float64(size) / elapsed.Seconds()
		fmt.Printf("  UUID:  %8.0f IDs/sec\n", rate)
		fmt.Println()
	}
}

func runConcurrentBenchmark() {
	fmt.Println("Concurrent Generation")
	fmt.Println("--------------------")

	const workers = 4
	const perWorker = 50000

	// TSCID test
	g := tscid.NewGenerator()
	start := time.Now()

	done := make(chan bool, workers)
	for i := 0; i < workers; i++ {
		go func() {
			for j := 0; j < perWorker; j++ {
				_, err := g.Generate()
				if err != nil {
					log.Printf("TSCID error: %v", err)
				}
			}
			done <- true
		}()
	}

	for i := 0; i < workers; i++ {
		<-done
	}

	tscidTime := time.Since(start)
	tscidRate := float64(workers*perWorker) / tscidTime.Seconds()

	// UUID test
	start = time.Now()

	for i := 0; i < workers; i++ {
		go func() {
			for j := 0; j < perWorker; j++ {
				_ = uuid.New()
			}
			done <- true
		}()
	}

	for i := 0; i < workers; i++ {
		<-done
	}

	uuidTime := time.Since(start)
	uuidRate := float64(workers*perWorker) / uuidTime.Seconds()

	fmt.Printf("TSCID: %8.0f IDs/sec (%d workers)\n", tscidRate, workers)
	fmt.Printf("UUID:  %8.0f IDs/sec (%d workers)\n", uuidRate, workers)
	fmt.Println()
}

func runOrderingTest() {
	fmt.Println("Temporal Ordering")
	fmt.Println("-----------------")

	const count = 5000
	ids := make([]tscid.TSCID, count)

	// Generate with small delays
	for i := 0; i < count; i++ {
		ids[i] = tscid.NewID()
		if i%100 == 0 && i > 0 {
			time.Sleep(time.Microsecond * 10)
		}
	}

	// Check if they're already sorted
	ordered := true
	for i := 1; i < len(ids); i++ {
		if ids[i-1].Compare(ids[i]) > 0 {
			ordered = false
			break
		}
	}

	fmt.Printf("Generated %d IDs\n", count)
	fmt.Printf("Naturally ordered: %v\n", ordered)

	// Test string sorting
	strings := make([]string, len(ids))
	for i, id := range ids {
		strings[i] = id.String()
	}

	// Shuffle
	rand.Shuffle(len(strings), func(i, j int) {
		strings[i], strings[j] = strings[j], strings[i]
	})

	start := time.Now()
	sort.Strings(strings)
	sortTime := time.Since(start)

	fmt.Printf("String sort time: %v\n", sortTime)
	fmt.Println()
}

func runStringBenchmark() {
	fmt.Println("String Operations")
	fmt.Println("-----------------")

	const iterations = 50000

	id := tscid.NewID()
	str := id.String()

	// Encoding
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_ = id.String()
	}
	encodeTime := time.Since(start)

	// Parsing
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, err := tscid.Parse(str)
		if err != nil {
			log.Printf("Parse error: %v", err)
		}
	}
	parseTime := time.Since(start)

	encodeRate := float64(iterations) / encodeTime.Seconds()
	parseRate := float64(iterations) / parseTime.Seconds()

	fmt.Printf("String encode: %8.0f ops/sec\n", encodeRate)
	fmt.Printf("String parse:  %8.0f ops/sec\n", parseRate)
	fmt.Printf("String length: %d chars\n", len(str))
	fmt.Println()
}

func runCollisionTest() {
	fmt.Println("Collision Test")
	fmt.Println("--------------")

	const count = 500000
	seen := make(map[string]bool, count)
	collisions := 0

	start := time.Now()
	for i := 0; i < count; i++ {
		id := tscid.NewID()
		str := id.String()

		if seen[str] {
			collisions++
		} else {
			seen[str] = true
		}
	}
	elapsed := time.Since(start)

	fmt.Printf("Generated: %d IDs in %v\n", count, elapsed)
	fmt.Printf("Collisions: %d\n", collisions)
	fmt.Printf("Unique rate: %.4f%%\n", float64(len(seen))/float64(count)*100)
	fmt.Println()
}

func runMemoryTest() {
	fmt.Println("Memory Usage")
	fmt.Println("------------")

	const count = 50000

	// Measure slice allocation
	sliceSize := count * 16 // Each TSCID is 16 bytes

	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	ids := make([]tscid.TSCID, count)
	for i := 0; i < count; i++ {
		ids[i] = tscid.NewID()
	}

	runtime.GC()
	runtime.ReadMemStats(&m2)

	// Calculate actual memory usage more accurately
	totalAlloc := m2.TotalAlloc - m1.TotalAlloc

	fmt.Printf("Generated: %d IDs\n", count)
	fmt.Printf("Slice size: %d bytes (%d bytes per ID)\n", sliceSize, 16)
	fmt.Printf("Total allocated: %d bytes\n", totalAlloc)
	fmt.Printf("Allocation overhead: ~%d bytes\n", totalAlloc-uint64(sliceSize))

	// Prevent optimization
	_ = ids[count-1]
	fmt.Println()
}

func runSortingComparison() {
	fmt.Println("Sorting Comparison")
	fmt.Println("------------------")

	const count = 50000

	// Generate data
	tscids := make([]string, count)
	uuids := make([]string, count)

	for i := 0; i < count; i++ {
		tscids[i] = tscid.NewID().String()
		uuids[i] = uuid.New().String()
	}

	// Shuffle both
	rand.Shuffle(len(tscids), func(i, j int) {
		tscids[i], tscids[j] = tscids[j], tscids[i]
	})
	rand.Shuffle(len(uuids), func(i, j int) {
		uuids[i], uuids[j] = uuids[j], uuids[i]
	})

	// Sort TSCID
	start := time.Now()
	sort.Strings(tscids)
	tscidSortTime := time.Since(start)

	// Sort UUID
	start = time.Now()
	sort.Strings(uuids)
	uuidSortTime := time.Since(start)

	fmt.Printf("TSCID sort: %v\n", tscidSortTime)
	fmt.Printf("UUID sort:  %v\n", uuidSortTime)

	if tscidSortTime < uuidSortTime {
		ratio := float64(uuidSortTime) / float64(tscidSortTime)
		fmt.Printf("TSCID %.1fx faster\n", ratio)
	} else {
		ratio := float64(tscidSortTime) / float64(uuidSortTime)
		fmt.Printf("UUID %.1fx faster\n", ratio)
	}

	fmt.Println()
	fmt.Println("Note: Sorting performance can vary based on data patterns")
	fmt.Println("and system characteristics. Results may differ between runs.")
}
