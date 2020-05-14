# profiler-get-symbols

The crates in this repo allow you to obtain symbol tables from ELF, Mach-O and PE
binaries as well as from pdb files. The implementation makes use of the crates
`object`, `goblin` and `pdb`.

The `lib` directory contains a generic Rust implementation. The `wasm` directory
contains a wrapper that targets WebAssembly and JavaScript.
The `examples` directory contains a command line tool that can be used to test
the functionality, for example as follows (executed from the workspace root):

```
cargo run -p dump-table -- firefox.pdb fixtures/win64-ci
```

The .wasm file and the JavaScript bindings are used by the Gecko profiler.
More specifically, they are used by the
[ProfilerGetSymbols.jsm](https://searchfox.org/mozilla-central/source/browser/components/extensions/ProfilerGetSymbols.jsm) module in Firefox. The code is run every time you use the Gecko profiler: On macOS and Linux
it is used to get symbols for native system libraries, and on all platforms it
is used if you're profiling a local build of Firefox for which there are no
symbols on the [Mozilla symbol server](https://symbols.mozilla.org/).


## Building

One-time setup:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli # --force to update
```

On changes:

```bash
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/profiler_get_symbols_wasm.wasm --out-dir . --no-modules --no-typescript
shasum -b -a 384 profiler_get_symbols_wasm_bg.wasm | awk '{ print $1 }' | xxd -r -p | base64 # This is your SRI hash, update it in index.html
```

## Running / Testing

This repo contains a minimal `index.html` which lets you test the resulting wasm
module manually in the browser. You can use the files in the `fixtures` directory
as examples.

To test, as a one-time setup, install http-server using cargo:

```bash
cargo install http-server
```

(The advantage of this over python's `SimpleHTTPServer` is that `http-server` sends the correct mime type for .wasm files.)

Then start the server in this directory, by typing `http-server` and pressing enter:

```bash
$ http-server
Starting up http-server, serving ./
Available on:
  http://0.0.0.0:8080
Hit CTRL-C to stop the server
```

Now you can open [http://0.0.0.0:8080](http://0.0.0.0:8080) in your browser and play with the API.

## Publishing

At the moment, the resulting wasm files are hosted in a separate repo called
[`profiler-assets`](https://github.com/mstange/profiler-assets/), in the
[`assets/wasm` directory](https://github.com/mstange/profiler-assets/tree/master/assets/wasm).
The filename of each of those wasm file is the same as its SRI hash value, but expressed in hexadecimal
instead of base64. Here's a command which creates a file with such a name from your `profiler_get_symbols_bg.wasm`:

```bash
cp profiler_get_symbols_wasm_bg.wasm `shasum -b -a 384 profiler_get_symbols_wasm_bg.wasm | awk '{ print $1 }'`.wasm
```
