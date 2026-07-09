module.exports = {
  extends: ["@commitlint/config-conventional"],
  rules: {
    "type-enum": [
      2,
      "always",
      [
        "feat",
        "fix",
        "perf",
        "refactor",
        "test",
        "bench",
        "docs",
        "ci",
        "chore",
        "revert",
      ],
    ],
    "scope-enum": [
      1,
      "always",
      [
        "harness",
        "engine",
        "metrics",
        "thermal",
        "mlflow",
        "kernels",
        "rust",
        "datasets",
        "analysis",
        "ci",
      ],
    ],
    "subject-max-length": [2, "always", 72],
  },
};
