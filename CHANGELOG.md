# Changelog

## [0.1.2](https://github.com/weklund/mlx-inference-workbench/compare/mlx-inference-workbench-v0.1.1...mlx-inference-workbench-v0.1.2) (2026-07-10)


### Features

* add agentic_coding_v1 hashed prompt dataset ([5ab44f2](https://github.com/weklund/mlx-inference-workbench/commit/5ab44f26f4df456d8cdaf45e92b265b094e83fe3))
* add agentic_coding_v1 hashed prompt dataset ([3e7bb0a](https://github.com/weklund/mlx-inference-workbench/commit/3e7bb0a32ee509ecd78eea4436ae571004179106)), closes [#6](https://github.com/weklund/mlx-inference-workbench/issues/6)
* add coverage gate for workbench scientific core ([3e91d84](https://github.com/weklund/mlx-inference-workbench/commit/3e91d84234e5b06f04585076b148a06f29e19002)), closes [#23](https://github.com/weklund/mlx-inference-workbench/issues/23)
* add m5 max ceilings via metal stream and mlx probes ([79b2e2f](https://github.com/weklund/mlx-inference-workbench/commit/79b2e2ff4de1befb32245aa351dca2a90fb3040d)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* coverage gate for workbench scientific core ([b5e17ca](https://github.com/weklund/mlx-inference-workbench/commit/b5e17caa7a6afd6949a850e180d129c0c33fddbb))
* dataset references for correctness gate ([4fe374f](https://github.com/weklund/mlx-inference-workbench/commit/4fe374f86163d066f4bd67a02cd68acf962ffea4))
* **harness:** phase 1 mvp measurement core ([d767f1c](https://github.com/weklund/mlx-inference-workbench/commit/d767f1c3ab17f63965745835e2a7e76743f2829a))
* m5 max ceilings via metal stream and mlx probes ([#8](https://github.com/weklund/mlx-inference-workbench/issues/8)) ([b4dfa3b](https://github.com/weklund/mlx-inference-workbench/commit/b4dfa3bf406e715b3ce75efbb56a6b4fd87ad14e))
* mlx-lm provisional baseline path ([#7](https://github.com/weklund/mlx-inference-workbench/issues/7)) ([deb8b1a](https://github.com/weklund/mlx-inference-workbench/commit/deb8b1a523b8ad1795c176b23f75ffcc767935fb))
* mlx-lm seed/memory polish and provisional baseline path ([8387683](https://github.com/weklund/mlx-inference-workbench/commit/83876836ca8ffca579bb88027a5740b2c7e699e1)), closes [#7](https://github.com/weklund/mlx-inference-workbench/issues/7)
* optional system_prompt on dataset items through generation ([0c37520](https://github.com/weklund/mlx-inference-workbench/commit/0c375209120dcce58facdc23868a795d9ee07c7a))
* wire correctness gate to dataset reference outputs ([8dacab9](https://github.com/weklund/mlx-inference-workbench/commit/8dacab9d8775372d036b63187665dbb6a4d0cfd1))


### Bug Fixes

* document non-macos metal_stream stubs for CI ([db93a88](https://github.com/weklund/mlx-inference-workbench/commit/db93a88d5df039e60577c709d92d4d854c7cb910))
* format prompts.py for ci ruff check ([8e269ad](https://github.com/weklund/mlx-inference-workbench/commit/8e269adf470bb299376e5f70c02cc5f39722cf15))
* harden harness CLI, gates, timeout, and store paths ([c7f42ed](https://github.com/weklund/mlx-inference-workbench/commit/c7f42ed87ffdde6b9f05e2a5a20175fca0090d36))
* keep coderabbit tone_instructions within 250 chars ([34fb106](https://github.com/weklund/mlx-inference-workbench/commit/34fb10658766585b893d284ad3e818db3c504890)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* limit stream TypeError fallback to call-time only ([9fa8dc3](https://github.com/weklund/mlx-inference-workbench/commit/9fa8dc3ea5b94e2fefa42c74fe78d3914a388675))
* load engine builtins even after custom registration ([3081739](https://github.com/weklund/mlx-inference-workbench/commit/308173932cd5146688b48b7e6c8cf6457a3d590f))
* resolve project root without package install path ([b7a0b52](https://github.com/weklund/mlx-inference-workbench/commit/b7a0b520f9eb04e5e0ac5e05fb760a3a77714443))
* route all experiment config flags through _parse_bool ([ffb2f0a](https://github.com/weklund/mlx-inference-workbench/commit/ffb2f0a60a0e51535707e4b6a47262f8434df94e))
* route all experiment config flags through _parse_bool ([76604f6](https://github.com/weklund/mlx-inference-workbench/commit/76604f6b0edd181b188af8701dbfb9e91c86d148)), closes [#21](https://github.com/weklund/mlx-inference-workbench/issues/21)
* run timed generate on a daemon thread ([4f24507](https://github.com/weklund/mlx-inference-workbench/commit/4f24507be4975f35db2a2cf4bfdbf254e3d5bbd2))
* treat non-stream generation as e2e-only metrics ([831476b](https://github.com/weklund/mlx-inference-workbench/commit/831476b81e42351af91ce8c700b6c29389fd369a))


### Refactoring

* give orchestrator sole ownership of timed generation ([36db1a9](https://github.com/weklund/mlx-inference-workbench/commit/36db1a9b335d3e3aa7112091e95c319739013a5d))
* orchestrator owns timed generation ([0b205c9](https://github.com/weklund/mlx-inference-workbench/commit/0b205c9b3344445ed3795609d030a2e247683d07))
* put note_duration on ThermalSensor protocol ([cc23ce0](https://github.com/weklund/mlx-inference-workbench/commit/cc23ce090c2f7064156bc2d778be6874e8847dc2))
* put note_duration on ThermalSensor protocol ([a85058b](https://github.com/weklund/mlx-inference-workbench/commit/a85058b9313e9f8a9109edd51f084c221bbd04e6)), closes [#20](https://github.com/weklund/mlx-inference-workbench/issues/20)
* single allowlist for distribution metric field names ([2ccefa6](https://github.com/weklund/mlx-inference-workbench/commit/2ccefa68e807825b0fab82ee7b042b3039875036))
* single allowlist for distribution metric field names ([0785dd3](https://github.com/weklund/mlx-inference-workbench/commit/0785dd3d01bb05f12807ec21c1d84ccc8be5d6b5)), closes [#22](https://github.com/weklund/mlx-inference-workbench/issues/22)


### Documentation

* align engine contract with e2e-only empty timestamps ([c185e0a](https://github.com/weklund/mlx-inference-workbench/commit/c185e0a88d57402b6e8aaece0478fa4ade8aa429))
* finish Phase 1 TASKS residual cleanup after MVP audit ([fcaf759](https://github.com/weklund/mlx-inference-workbench/commit/fcaf759447ec15ad1b800a3cd06a708cd05b67a1))
* mark Phase 1 MVP harness done in TASKS ([#24](https://github.com/weklund/mlx-inference-workbench/issues/24) audit) ([e2f638b](https://github.com/weklund/mlx-inference-workbench/commit/e2f638b428840490936c9180de10d9500d11b419))
* mark Phase 1 MVP harness done in TASKS after [#24](https://github.com/weklund/mlx-inference-workbench/issues/24) audit ([6ef62e4](https://github.com/weklund/mlx-inference-workbench/commit/6ef62e480b617580c9a5fe40e70b5ccff84bf74c))
* restore thermal evening cohort and add day-2 morning data ([8a9fb2a](https://github.com/weklund/mlx-inference-workbench/commit/8a9fb2aa6a3b879ddb259e9e75d35f9f3db9b422))
* thermal day-2 morning + restored battery evening cohort ([e066296](https://github.com/weklund/mlx-inference-workbench/commit/e066296d4733d7bb1139d583bb2cfbf2dfbac304))


### CI/CD

* add assertive coderabbit config for scientific workbench ([3bc3fd9](https://github.com/weklund/mlx-inference-workbench/commit/3bc3fd92b4705d1dbc8674ee6b3121a519ac0bbc)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* add Makefile as single entrypoint for CI and local tasks ([2cbeb9a](https://github.com/weklund/mlx-inference-workbench/commit/2cbeb9a2493adbf6fdbb44ed7619fe5b6fce3cc9))
* require Test + Coverage check to merge into main ([2edf46b](https://github.com/weklund/mlx-inference-workbench/commit/2edf46b111de36438f57ee60cbd4b800495bd69b))
* run rust lint on macos for metal path coverage ([75e0a7e](https://github.com/weklund/mlx-inference-workbench/commit/75e0a7ebc32d38c9532617ac08fd21d314c4eb66))
* set workflow GITHUB_TOKEN to contents read-only ([babcd2d](https://github.com/weklund/mlx-inference-workbench/commit/babcd2df5d743f3bc04568edfc04a22fc5d35fba)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* skip cargo-semver-checks for unpublished workspace crates ([d9d6115](https://github.com/weklund/mlx-inference-workbench/commit/d9d6115103d8f2133f6313942830af578729d914)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* skip semver-checks for packages missing at baseline ([5deb29a](https://github.com/weklund/mlx-inference-workbench/commit/5deb29a7aa647276142ff49911237332e404fe24)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)

## [0.1.1](https://github.com/weklund/mlx-inference-workbench/compare/mlx-inference-workbench-v0.1.0...mlx-inference-workbench-v0.1.1) (2026-07-09)


### Features

* **harness:** add Phase 0.5 thermal validation tooling + initial data ([ac99114](https://github.com/weklund/mlx-inference-workbench/commit/ac9911457bfa868ee7f43753a650602dc539e529))
* **harness:** add power gate, fix thermal sensors, clean validation data ([f251f97](https://github.com/weklund/mlx-inference-workbench/commit/f251f971a0b5f2f9e181d5a2953503c62bf863ea))
* **harness:** power mode detection + valid thermal data ([d7ff53c](https://github.com/weklund/mlx-inference-workbench/commit/d7ff53c6031a730b8a8eb450a671414a1ebf8a7b))
* initialize mlx-inference-workbench repo ([9d70d6a](https://github.com/weklund/mlx-inference-workbench/commit/9d70d6a9929f333c5f08252acf2c1f9b3edefd9b))


### Bug Fixes

* apply ruff formatting ([c38058e](https://github.com/weklund/mlx-inference-workbench/commit/c38058ed96a972dc5b057b117d727493e9ef3b12))
* **ci:** add jsonpath to CITATION.cff extra-files entry ([9151672](https://github.com/weklund/mlx-inference-workbench/commit/915167267bd63db679b4e2889f96c07de7488d8b))
* **ci:** add uv.lock and fix deprecated dev-dependencies ([d646e02](https://github.com/weklund/mlx-inference-workbench/commit/d646e02efa552e0a4c730427903ed70376e9736b))
* **ci:** allow integration test job to pass with no tests collected ([02debef](https://github.com/weklund/mlx-inference-workbench/commit/02debefb0dab04a36cd3a067967d965597249d2d))
* **ci:** remove bare pyproject.toml from release-please extra-files ([63ccf32](https://github.com/weklund/mlx-inference-workbench/commit/63ccf32bd7facb7fa092e1867f6f4ed9f95da1ca))
* **ci:** resolve workflow file errors ([a0498bb](https://github.com/weklund/mlx-inference-workbench/commit/a0498bbf4c425a445cb00f720b867189de2cbf94))
* remove extraneous f-string prefixes (ruff F541) ([f522ac4](https://github.com/weklund/mlx-inference-workbench/commit/f522ac46cac5c45374c29bd84ea55f4714c76579))


### Documentation

* add CITATION.cff for software citation ([779dfb1](https://github.com/weklund/mlx-inference-workbench/commit/779dfb14b7a73c518e32bffccdcc82e44073caf1))
* fix citation name and add citing section to README ([8f0270f](https://github.com/weklund/mlx-inference-workbench/commit/8f0270f151725fb496732f99496b188662feed08))
* **harness:** complete Phase 0 MTPLX familiarization spike ([764c148](https://github.com/weklund/mlx-inference-workbench/commit/764c148d507fe505511ac6e50953f073906666e0))


### CI/CD

* auto-update CITATION.cff version on release ([d42134d](https://github.com/weklund/mlx-inference-workbench/commit/d42134d52ce8ab0ccabc3ccbf625a9af0c880f7f))


### Tests

* add placeholder test so pytest doesn't exit 5 on empty collection ([5319780](https://github.com/weklund/mlx-inference-workbench/commit/531978022fb38620451a3b9be0b2526e86f23761))
