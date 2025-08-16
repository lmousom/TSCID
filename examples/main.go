package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/lmousom/tscid"
)

func basicUsage() {
	fmt.Println("Basic usage:")
	id := tscid.NewID()
	fmt.Printf("Generated: %s\n", id)
	fmt.Printf("Created at: %v\n", id.Time())
	fmt.Printf("Age: %v\n\n", id.Age())
}

func batchGeneration() {
	fmt.Println("Batch generation:")
	ids := tscid.NewBatch(3)
	for _, id := range ids {
		fmt.Printf("%s\n", id.Compact())
	}
	fmt.Println()
}

func jsonExample() {
	fmt.Println("JSON marshaling:")

	type User struct {
		ID   tscid.TSCID `json:"id"`
		Name string      `json:"name"`
	}

	user := User{ID: tscid.NewID(), Name: "john"}
	data, _ := json.Marshal(user)
	fmt.Printf("%s\n", data)

	var u User
	json.Unmarshal(data, &u)
	fmt.Printf("Parsed ID: %s\n\n", u.ID)
}

func ordering() {
	fmt.Println("Temporal ordering:")

	first := tscid.NewID()
	time.Sleep(5 * time.Millisecond)
	second := tscid.NewID()

	fmt.Printf("First:  %s\n", first.Compact())
	fmt.Printf("Second: %s\n", second.Compact())
	fmt.Printf("First < Second: %v\n\n", first.Less(second))
}

func autoMode() {
	fmt.Println("Auto mode performance:")

	gen := tscid.NewSmartGenerator()

	// Single-threaded
	start := time.Now()
	for i := 0; i < 1000; i++ {
		gen.Generate()
	}
	single := time.Since(start)

	// Multi-threaded
	start = time.Now()
	done := make(chan bool, 4)
	for i := 0; i < 4; i++ {
		go func() {
			for j := 0; j < 250; j++ {
				gen.Generate()
			}
			done <- true
		}()
	}
	for i := 0; i < 4; i++ {
		<-done
	}
	multi := time.Since(start)

	fmt.Printf("Single-threaded: %v\n", single)
	fmt.Printf("Multi-threaded:  %v\n", multi)
	fmt.Println("Same generator adapted automatically")
}

func generators() {
	fmt.Println("Different generators:")

	safe := tscid.NewGenerator()
	fast := tscid.NewFastGenerator()
	smart := tscid.NewSmartGenerator()

	safeID, _ := safe.Generate()
	fastID, _ := fast.Generate()
	smartID, _ := smart.Generate()

	fmt.Printf("Safe:  %s\n", safeID.Compact())
	fmt.Printf("Fast:  %s\n", fastID.Compact())
	fmt.Printf("Smart: %s\n", smartID.Compact())
	fmt.Println()
}

func main() {
	fmt.Println("TSCID Examples")

	basicUsage()
	batchGeneration()
	jsonExample()
	ordering()
	autoMode()
	generators()

	fmt.Println("Database usage:")
	fmt.Println("type Product struct {")
	fmt.Println("    ID tscid.TSCID `db:\"id\"`")
	fmt.Println("    Name string `db:\"name\"`")
	fmt.Println("}")
}
