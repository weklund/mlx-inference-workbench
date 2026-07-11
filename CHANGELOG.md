# Changelog

## [0.1.3](https://github.com/weklund/mlx-inference-workbench/compare/mlx-inference-workbench-v0.1.2...mlx-inference-workbench-v0.1.3) (2026-07-11)


### Features

* add agentic_coding_v1 hashed prompt dataset ([19d2b3f](https://github.com/weklund/mlx-inference-workbench/commit/19d2b3f4fbdb232e4f41c02d0d047c9ef34aeee1))
* add agentic_coding_v1 hashed prompt dataset ([a5f4839](https://github.com/weklund/mlx-inference-workbench/commit/a5f4839a280541642b100d58bd57f9a3df37f792)), closes [#6](https://github.com/weklund/mlx-inference-workbench/issues/6)
* add coverage gate for workbench scientific core ([e10d6a6](https://github.com/weklund/mlx-inference-workbench/commit/e10d6a674721128e8de3c1407534c7f82d14e7fb)), closes [#23](https://github.com/weklund/mlx-inference-workbench/issues/23)
* add m5 max ceilings via metal stream and mlx probes ([31bc742](https://github.com/weklund/mlx-inference-workbench/commit/31bc7427de383b9bf4a36c5258d724eceedc2f59)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* add MTPLX engine plugin with nested model.mtplx options ([6273ebf](https://github.com/weklund/mlx-inference-workbench/commit/6273ebf72614e8db5c7e9dddbc86456be36e5857))
* add MTPLX smoke config and HF model path resolve ([099fe11](https://github.com/weklund/mlx-inference-workbench/commit/099fe112af48b27e758bfc2d4156e210956acef1))
* coverage gate for workbench scientific core ([4a08e7b](https://github.com/weklund/mlx-inference-workbench/commit/4a08e7bdd439f833aa2573912d9c03eee3af3d6a))
* dataset references for correctness gate ([09f159d](https://github.com/weklund/mlx-inference-workbench/commit/09f159d76f63844b6db2d2a717a50b2d9a201169))
* **harness:** add Phase 0.5 thermal validation tooling + initial data ([a4af1d9](https://github.com/weklund/mlx-inference-workbench/commit/a4af1d9720ad2326a5ac0130cd31921912e349d8))
* **harness:** add phase 1 mvp measurement core with behavioral tests ([af80690](https://github.com/weklund/mlx-inference-workbench/commit/af806902722f4584f81bf7b93a79d4ff11cae77e))
* **harness:** add power gate, fix thermal sensors, clean validation data ([f2be193](https://github.com/weklund/mlx-inference-workbench/commit/f2be193fc03845e54e3db0532893fa999863affa))
* **harness:** phase 1 mvp measurement core ([319f123](https://github.com/weklund/mlx-inference-workbench/commit/319f12348809ec1fa23373c41672de0e6db1f83b))
* **harness:** power mode detection + valid thermal data ([05cd122](https://github.com/weklund/mlx-inference-workbench/commit/05cd12225c037b2ad2fdcfc453765237028aef0d))
* initialize mlx-inference-workbench repo ([b6d6393](https://github.com/weklund/mlx-inference-workbench/commit/b6d6393a49ccf9fe3c0dd9aedb15e91f0df5d742))
* m5 max ceilings via metal stream and mlx probes ([#8](https://github.com/weklund/mlx-inference-workbench/issues/8)) ([6fd0cb8](https://github.com/weklund/mlx-inference-workbench/commit/6fd0cb8df8ceb095240df28d9b2fd0e61425cab3))
* mlx-lm provisional baseline path ([#7](https://github.com/weklund/mlx-inference-workbench/issues/7)) ([5793762](https://github.com/weklund/mlx-inference-workbench/commit/5793762e3a203a37243b4ce87740d1bf0724d00b))
* mlx-lm seed/memory polish and provisional baseline path ([271954c](https://github.com/weklund/mlx-inference-workbench/commit/271954ce847c858eb0a88d12ce6bf1714849c63b)), closes [#7](https://github.com/weklund/mlx-inference-workbench/issues/7)
* MTPLX engine plugin + smoke (issue [#9](https://github.com/weklund/mlx-inference-workbench/issues/9)) ([34bba43](https://github.com/weklund/mlx-inference-workbench/commit/34bba434cdb86a1945cfa96100d145bd0bb01b4b))
* official mlx-lm baseline under full thermal protocol ([#36](https://github.com/weklund/mlx-inference-workbench/issues/36)) ([1a99f03](https://github.com/weklund/mlx-inference-workbench/commit/1a99f03916ebc45e4a49a84a10fc541e3e73ffd7))
* official mlx-lm baseline under thermal protocol ([#36](https://github.com/weklund/mlx-inference-workbench/issues/36)) ([056cf0e](https://github.com/weklund/mlx-inference-workbench/commit/056cf0e592f12acd44f73f9684a347f729762d84))
* optional system_prompt on dataset items through generation ([6eca5ee](https://github.com/weklund/mlx-inference-workbench/commit/6eca5ee37e22d7662a6cea50b13ef792c2c0c680))
* wire correctness gate to dataset reference outputs ([d8bb99e](https://github.com/weklund/mlx-inference-workbench/commit/d8bb99e588e2fdb1adf3cc640ad32d5d8375c5ec))


### Bug Fixes

* apply ruff formatting ([0a2d625](https://github.com/weklund/mlx-inference-workbench/commit/0a2d625f7eecafbc241c285e59cc04297f0eea4f))
* **ci:** add jsonpath to CITATION.cff extra-files entry ([2a36211](https://github.com/weklund/mlx-inference-workbench/commit/2a36211358183db59184b7152b36cab9ce434864))
* **ci:** add uv.lock and fix deprecated dev-dependencies ([9189083](https://github.com/weklund/mlx-inference-workbench/commit/91890833c6f81cbf753e5e53250e15ae93b20770))
* **ci:** allow integration test job to pass with no tests collected ([a2da06b](https://github.com/weklund/mlx-inference-workbench/commit/a2da06ba23068bc7a7c1772e12b401ef8b4b7068))
* **ci:** remove bare pyproject.toml from release-please extra-files ([8c839b0](https://github.com/weklund/mlx-inference-workbench/commit/8c839b0e9deb9abe22706d293e686f3f7dcb2118))
* **ci:** resolve workflow file errors ([25c962e](https://github.com/weklund/mlx-inference-workbench/commit/25c962ecc8f7eadfdaa5b1fb35e4ab32da5cfd96))
* document non-macos metal_stream stubs for CI ([b8ad394](https://github.com/weklund/mlx-inference-workbench/commit/b8ad3944ec35115ecd96a5e23759a7201cb28d96))
* fail closed on MTPLX token-count encode errors ([dd5e333](https://github.com/weklund/mlx-inference-workbench/commit/dd5e3336e455ae1fa9d4d3c736d40452fa0b24be))
* fail closed when MTPLX output lacks text ([442c333](https://github.com/weklund/mlx-inference-workbench/commit/442c3331ee1c1509e594b20d1ab09c07a3e359e1))
* fail closed when powermetrics omits thermal pressure ([3479eb8](https://github.com/weklund/mlx-inference-workbench/commit/3479eb80b69cb4e075fca110e1f1241a88a3e63a))
* format prompts.py for ci ruff check ([a27c18b](https://github.com/weklund/mlx-inference-workbench/commit/a27c18b995eb09388294c389d668a9d0acd23b64))
* harden harness CLI, gates, timeout, and store paths ([fef1d10](https://github.com/weklund/mlx-inference-workbench/commit/fef1d108b3312f0521dc8a1c1bd5acbd3382abed))
* honest protocol baseline labeling and HLD warmup ([4720374](https://github.com/weklund/mlx-inference-workbench/commit/47203742890d733235bdfaf29217c90f8500277d))
* keep coderabbit tone_instructions within 250 chars ([3b95717](https://github.com/weklund/mlx-inference-workbench/commit/3b957170d89845c729a265bd14603dfc0e53bcd7)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* limit stream TypeError fallback to call-time only ([e87cf7b](https://github.com/weklund/mlx-inference-workbench/commit/e87cf7bd93ae91c64b68a0cabc02eab0adc9e1e9))
* load engine builtins even after custom registration ([495756a](https://github.com/weklund/mlx-inference-workbench/commit/495756a83477a460530a6506e50f09079f05be3f))
* register MtplxEngine without swallowing ImportError ([8fce4ad](https://github.com/weklund/mlx-inference-workbench/commit/8fce4ad49926cff917cc44f1ee6bcb2c8158ebd9))
* remove extraneous f-string prefixes (ruff F541) ([d2fe88d](https://github.com/weklund/mlx-inference-workbench/commit/d2fe88d5632226455ca4985da5cb97332d6e8d8c))
* resolve project root without package install path ([ed9fdc6](https://github.com/weklund/mlx-inference-workbench/commit/ed9fdc6cb48d66a1011c8a9ad2a2fa0f116c2f99))
* route all experiment config flags through _parse_bool ([edef7bf](https://github.com/weklund/mlx-inference-workbench/commit/edef7bfcf8d345e7bfb752c20632f96598900c5a))
* route all experiment config flags through _parse_bool ([f3d7ae5](https://github.com/weklund/mlx-inference-workbench/commit/f3d7ae53195aeed01c586c8e01708bbf4e9863f5)), closes [#21](https://github.com/weklund/mlx-inference-workbench/issues/21)
* run timed generate on a daemon thread ([98a0f86](https://github.com/weklund/mlx-inference-workbench/commit/98a0f863566cb9e7a8fed315af378b4633fbcbae))
* treat non-stream generation as e2e-only metrics ([6513d91](https://github.com/weklund/mlx-inference-workbench/commit/6513d91c853a3d83e2f4c3d762552a3e676e9ad9))


### Refactoring

* give orchestrator sole ownership of timed generation ([a7c022e](https://github.com/weklund/mlx-inference-workbench/commit/a7c022e2f84c7e1bf1b449046c2377ae267ff35c))
* orchestrator owns timed generation ([79d7bf1](https://github.com/weklund/mlx-inference-workbench/commit/79d7bf19b70300cbd622ef21c08538eb4b897c45))
* put note_duration on ThermalSensor protocol ([f447434](https://github.com/weklund/mlx-inference-workbench/commit/f4474348ee7058556449ec3bb3c2185c497ed93b))
* put note_duration on ThermalSensor protocol ([e5bd140](https://github.com/weklund/mlx-inference-workbench/commit/e5bd1409e55344fc608a0a8cbb4081142328ad72)), closes [#20](https://github.com/weklund/mlx-inference-workbench/issues/20)
* single allowlist for distribution metric field names ([d09ada0](https://github.com/weklund/mlx-inference-workbench/commit/d09ada056683aa2aabb07f10d158305b30fdca94))
* single allowlist for distribution metric field names ([485165f](https://github.com/weklund/mlx-inference-workbench/commit/485165f442eecf4e296139d6ac3209912cc0a1c2)), closes [#22](https://github.com/weklund/mlx-inference-workbench/issues/22)


### Documentation

* add CITATION.cff for software citation ([cd37d49](https://github.com/weklund/mlx-inference-workbench/commit/cd37d49332becf213950c70a3fc8bfe3c23ab31a))
* add clean day-2 evening thermal protocol runs ([405874d](https://github.com/weklund/mlx-inference-workbench/commit/405874d103e93afe06e4e07d031b4072859ed481))
* add day-2 afternoon thermal protocol runs ([d25c7bf](https://github.com/weklund/mlx-inference-workbench/commit/d25c7bf32de907f703da6002c421ce28defa7dce))
* add day-2 evening thermal runs (GPU contention, excluded) ([4a910e6](https://github.com/weklund/mlx-inference-workbench/commit/4a910e6da48aec7e84590e7cbf8d1a7946560853))
* align engine contract with e2e-only empty timestamps ([343fb5c](https://github.com/weklund/mlx-inference-workbench/commit/343fb5c0f914efed86a972d6a23f6bd29f153871))
* align TASKS after thermal gate ([#3](https://github.com/weklund/mlx-inference-workbench/issues/3)) and MTPLX plugin ([6c331f7](https://github.com/weklund/mlx-inference-workbench/commit/6c331f7304e0ae82d7cbe4443474118d8e6aec5a))
* align TASKS roadmap after thermal gate and MTPLX plugin ([5981d50](https://github.com/weklund/mlx-inference-workbench/commit/5981d5002cbe7a04d7890b7ad41c2cc8ae4d7491))
* align TASKS wording with protocol-baseline honesty ([6e5acef](https://github.com/weklund/mlx-inference-workbench/commit/6e5acef30a7e8cd74c255eb0df6167c8d60b5cdf))
* clarify five-prompt pilot vs max_prompts load cap ([f1d833d](https://github.com/weklund/mlx-inference-workbench/commit/f1d833dd37b55eb7e8cf39a0c6146124808c6bb1))
* finish Phase 1 TASKS residual cleanup after MVP audit ([22949d2](https://github.com/weklund/mlx-inference-workbench/commit/22949d2d092072c8f4e471098ba964f0acbd1132))
* fix citation name and add citing section to README ([400a231](https://github.com/weklund/mlx-inference-workbench/commit/400a231e23ccf3753bf862eef9f05de2f633f291))
* **harness:** complete Phase 0 MTPLX familiarization spike ([c285376](https://github.com/weklund/mlx-inference-workbench/commit/c2853761944a8a14f60747a5d6a0897ff7090491))
* mark Phase 1 MVP harness done in TASKS ([#24](https://github.com/weklund/mlx-inference-workbench/issues/24) audit) ([ca89d30](https://github.com/weklund/mlx-inference-workbench/commit/ca89d306a6ff47bfb604f30d60a4e82933e642f8))
* mark Phase 1 MVP harness done in TASKS after [#24](https://github.com/weklund/mlx-inference-workbench/issues/24) audit ([6d82b53](https://github.com/weklund/mlx-inference-workbench/commit/6d82b53e8eca16d2f2155b06e26a3067f1831972))
* Phase 0.5 thermal data + reproducibility report ([#3](https://github.com/weklund/mlx-inference-workbench/issues/3)) ([29c0a69](https://github.com/weklund/mlx-inference-workbench/commit/29c0a69ebce3aabc274ebc2a7fa0f155c7baa835))
* restore thermal evening cohort and add day-2 morning data ([01e56a5](https://github.com/weklund/mlx-inference-workbench/commit/01e56a59cf17b32b9eeb336105c10fd092a4b636))
* thermal day-2 morning + restored battery evening cohort ([723b903](https://github.com/weklund/mlx-inference-workbench/commit/723b90343b3a59a4da3291c543e08ef16aa6b7ea))
* thermal reproducibility report and HLD §22 update ([#3](https://github.com/weklund/mlx-inference-workbench/issues/3)) ([54a704e](https://github.com/weklund/mlx-inference-workbench/commit/54a704e482e78bb5a637e3ba08e1438fdf6ba6f7))


### CI/CD

* add assertive coderabbit config for scientific workbench ([fecec29](https://github.com/weklund/mlx-inference-workbench/commit/fecec299fc85e99533879bf151d8491b9903b2fc)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* add Makefile as single entrypoint for CI and local tasks ([c244466](https://github.com/weklund/mlx-inference-workbench/commit/c2444663727709fe61cc0ebcbac825b156e17e22))
* auto-update CITATION.cff version on release ([226618d](https://github.com/weklund/mlx-inference-workbench/commit/226618d1773ae5d3342d44dc6862982bbe8c940b))
* require Test + Coverage check to merge into main ([369b0ab](https://github.com/weklund/mlx-inference-workbench/commit/369b0ab348120c2fff3cf4d319ba9875ae47aeec))
* run rust lint on macos for metal path coverage ([3c5eaed](https://github.com/weklund/mlx-inference-workbench/commit/3c5eaeddc72a55f969af628a4ea172f5e5c077b1))
* set workflow GITHUB_TOKEN to contents read-only ([a361643](https://github.com/weklund/mlx-inference-workbench/commit/a36164311cad138712840f3835f8a5d10ee5fb5d)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* skip cargo-semver-checks for unpublished workspace crates ([ec4cb57](https://github.com/weklund/mlx-inference-workbench/commit/ec4cb5743911dc5e133acc9460ab8dfcb6d2ebe1)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)
* skip semver-checks for packages missing at baseline ([351d0f2](https://github.com/weklund/mlx-inference-workbench/commit/351d0f248e4eb2f68f275e604a3b028f003a9485)), closes [#8](https://github.com/weklund/mlx-inference-workbench/issues/8)


### Tests

* add placeholder test so pytest doesn't exit 5 on empty collection ([e259a33](https://github.com/weklund/mlx-inference-workbench/commit/e259a338272732340e76b8188360dcd90ef72a20))

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
