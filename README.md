# Sojourn

A constrained random vector generator: give it a box and a set of constraints, and it
produces random points that satisfy them, by sampling, by walking, and by asking Z3 (the
`cvg` module). The constraints are written in a small expression language, babel:
`x1 + x2 * cos(x3)^2` is a transform, `x1 < x2 + x3` is a constraint. The crate parses
babel, evaluates it in batches, and reads its structure to search for feasible points.

```rust
let compiled = sojourn::compile("x1 + x2 > 20 - x3^2", &["x1", "x2", "x3"])?;

// One column per sample, one row per variable, in the order given.
let residuals = compiled.eval(samples.as_ref())?;
```

Source text goes in and nothing hands back a syntax tree: `compile` parses and binds an
expression to evaluate, `ConstraintSystem::new` parses a set of constraints to solve, and the
`FeasibleRegion` a solve returns hands out samples and repairs a point onto the region.

Sojourn is consumed as a cargo git dependency; it is not on crates.io.

## Building

Everything is a [`just`](https://github.com/casey/just) recipe, and CI runs exactly `just ci`.

```
just build          compile the crate and every test target
just test           run the test suite with nextest
just lint           rustfmt and clippy, warnings denied
```

The lexer and parser are generated from [`grammar/`](grammar) by `build.rs` at build time.
The GPU sieve is behind the `gpu` feature and off by default.

## Reading further

- [`src/README.md`](src/README.md): the architecture, one AST and two backends.
- [`AGENTS.md`](AGENTS.md): conventions, layout, and the reading order for the design notes.
- [`docs/todo.md`](docs/todo.md): the roadmap and the reasoning behind the decisions.

## License

Apache-2.0. See [LICENSE](LICENSE). The constrained random vector generator descends from
the Apache-2 [sojourn-CVG](https://github.com/Groostav/sojourn-CVG) project; its original
notes are under [`docs/sojourn/`](docs/sojourn).
