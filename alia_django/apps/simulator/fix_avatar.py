import re

# Read modeling HTML to extract just the ThreeJS logic (init, animate, etc.)
mod_path = r'c:\Users\MSI\Downloads\Projet DS\alia_django\templates\modeling\index.html'
with open(mod_path, 'r', encoding='utf-8') as f:
    mod_html = f.read()

# I will just write a custom JS module for simulator that contains the exact Three.js logic from modeling
avatar_script = """
<script type="module">
    import * as THREE from 'three';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
    import { avatarKeyframes } from '{% static "avatar-animation.js" %}';

    const AVATAR_GLB_URL = '/alia-api/static/avatar.glb';

    const avatar = {
        model: null, mixer: null,
        neck: null, head: null,
        leftShoulder: null, rightShoulder: null,
        leftArm: null, rightArm: null,
        leftForeArm: null, rightForeArm: null,
        spine1: null, spine2: null,
        jawMeshes: [], visemeMeshes: [], blinkMeshes: [], eyeLookMeshes: [],
        rest: {}, loaded: false
    };

    const clock = new THREE.Clock();
    const LIP_FFT = 512;
    const audioData = new Uint8Array(LIP_FFT / 2);
    const timeData = new Uint8Array(LIP_FFT);
    const LIPEASE_VOWEL = 0.32;
    const LIPEASE_FRIC = 0.38;
    const LIPEASE_PLOSIVE = 0.52;
    const LIP_DECAY = 0.2;
    const LIP_ORDER = ['aa', 'O', 'E', 'I', 'U', 'CH', 'SS', 'FF', 'PP', 'TH', 'DD', 'kk', 'nn', 'RR', 'sil'];
    const smoothVis = Object.fromEntries(LIP_ORDER.map((k) => [k, k === 'sil' ? 1 : 0]));
    
    let scene, camera, renderer, analyzer, audioCtx;
    let animFrameId = null;
    let isSpeaking = false;
    let gazeDirX = 0, gazeDirY = 0, gazeTargetX = 0, gazeTargetY = 0, gazeTimer = 0;
    let smoothIntensity = 0, prevEnergy = 0, prevRmsAmp = 0, armWeight = 0;
    let t_anim = 0, bodyAnimT = 0;
    let containerReady = false;

    window._initAvatar3D = function() {
        const container = document.getElementById('avatar3dContainer');
        if (!container || containerReady) return;
        containerReady = true;

        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(28, 1, 0.1, 1000);
        camera.position.set(0, 1.55, 2.5);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        function refitCanvas() {
            if (!container || !renderer || !camera) return;
            const w = Math.max(1, container.clientWidth);
            const h = Math.max(1, container.clientHeight);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }
        refitCanvas();
        container.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.75));
        const key = new THREE.DirectionalLight(0xffffff, 1.1);
        key.position.set(1, 2.5, 2);
        const fill = new THREE.DirectionalLight(0x8bb8ff, 0.35);
        fill.position.set(-2, 1, 1);
        scene.add(key, fill);

        new GLTFLoader().load(AVATAR_GLB_URL, (gltf) => {
            avatar.model = gltf.scene;
            avatar.model.rotation.y = Math.PI + 10;
            avatar.model.rotation.y += 0.15;
            scene.add(avatar.model);

            if (gltf.animations.length > 0) {
                avatar.mixer = new THREE.AnimationMixer(avatar.model);
                const action = avatar.mixer.clipAction(gltf.animations[0]);
                action.setLoop(THREE.LoopRepeat, Infinity);
                action.play();
            }

            avatar.model.traverse((child) => {
                if (child.isBone) {
                    switch (child.name) {
                        case 'Neck': avatar.neck = child; break;
                        case 'Head': avatar.head = child; break;
                        case 'LeftShoulder': avatar.leftShoulder = child; break;
                        case 'RightShoulder': avatar.rightShoulder = child; break;
                        case 'LeftArm': avatar.leftArm = child; break;
                        case 'RightArm': avatar.rightArm = child; break;
                        case 'LeftForeArm': avatar.leftForeArm = child; break;
                        case 'RightForeArm': avatar.rightForeArm = child; break;
                        case 'Spine1': avatar.spine1 = child; break;
                        case 'Spine2': avatar.spine2 = child; break;
                    }
                }
                if (child.isSkinnedMesh && child.morphTargetDictionary) {
                    const d = child.morphTargetDictionary;
                    if (d['jawOpen'] !== undefined) {
                        avatar.jawMeshes.push({ mesh: child, jawOpen: d['jawOpen'], mouthOpen: d['mouthOpen'] ?? null });
                    }
                    if (d['viseme_aa'] !== undefined) {
                        const pairs = [['aa','viseme_aa'],['O','viseme_O'],['E','viseme_E'],['I','viseme_I'],['U','viseme_U'],
                            ['CH','viseme_CH'],['SS','viseme_SS'],['FF','viseme_FF'],['PP','viseme_PP'],
                            ['TH','viseme_TH'],['DD','viseme_DD'],['kk','viseme_kk'],['nn','viseme_nn'],
                            ['RR','viseme_RR'],['sil','viseme_sil']];
                        const vis = {};
                        for (const [key, morph] of pairs) { if (d[morph] !== undefined) vis[key] = d[morph]; }
                        avatar.visemeMeshes.push({ mesh: child, vis });
                    }
                }
            });

            function saveRest(name, bone) {
                if (!bone) return;
                avatar.rest[name] = { x: bone.rotation.x, y: bone.rotation.y, z: bone.rotation.z };
            }
            saveRest("leftArm", avatar.leftArm);
            saveRest("rightArm", avatar.rightArm);
            saveRest("leftForeArm", avatar.leftForeArm);
            saveRest("rightForeArm", avatar.rightForeArm);
            saveRest("spine2", avatar.spine2);
            saveRest("neck", avatar.neck);

            applyRestPose();
            refitCanvas();
            requestAnimationFrame(() => { refitCanvas(); requestAnimationFrame(refitCanvas); });
            avatar.loaded = true;
        });

        animate();
        new ResizeObserver(refitCanvas).observe(container);
        window.addEventListener('resize', refitCanvas, { passive: true });
    };

    function applyRestPose() {
        if (avatar.leftShoulder) avatar.leftShoulder.rotation.set(0.1, 0, 0.28);
        if (avatar.rightShoulder) avatar.rightShoulder.rotation.set(0.1, 0, -0.28);
        if (avatar.leftArm) avatar.leftArm.rotation.set(0, 0, 0.75);
        if (avatar.rightArm) avatar.rightArm.rotation.set(0, 0, -0.75);
        if (avatar.spine2) avatar.spine2.rotation.x = 0.03;
    }

    function resetLipState() {
        smoothIntensity = 0; prevEnergy = 0; prevRmsAmp = 0;
        for (const k of LIP_ORDER) smoothVis[k] = k === 'sil' ? 1 : 0;
    }

    function bandMean(start, end) {
        let s = 0; const e = Math.min(end, audioData.length);
        const n = Math.max(1, e - start);
        for (let i = start; i < e; i++) s += audioData[i] / 255;
        return s / n;
    }

    function paintVisemeMeshes(meshBlend = 0.38) {
        const keyScale = { CH: 0.9, SS: 0.85, FF: 0.8, TH: 0.88, PP: 0.98, DD: 0.94, kk: 0.92, nn: 0.78, RR: 0.82 };
        for (const vm of avatar.visemeMeshes) {
            const inf = vm.mesh.morphTargetInfluences;
            for (const k of LIP_ORDER) {
                const idx = vm.vis[k];
                if (idx === undefined) continue;
                const t = smoothVis[k] * (keyScale[k] ?? 1);
                inf[idx] = THREE.MathUtils.lerp(inf[idx], t, meshBlend);
            }
        }
    }

    function smoothTowardVis(target, rates) {
        for (const k of LIP_ORDER) {
            const r = rates[k] ?? LIPEASE_VOWEL;
            smoothVis[k] = THREE.MathUtils.lerp(smoothVis[k], target[k], r);
        }
    }

    function normalizeCompetitive(weights) {
        let sum = 0;
        for (const k in weights) { const w = Math.max(0, weights[k]); weights[k] = w; sum += w; }
        if (sum <= 0.00001) return weights;
        for (const k in weights) weights[k] /= sum;
        return weights;
    }

    function animate() {
        animFrameId = requestAnimationFrame(animate);
        const time = performance.now() * 0.001;
        const delta = clock.getDelta();

        if (avatar.mixer) avatar.mixer.update(delta);
        if (!avatar.model) { if(renderer && scene && camera) renderer.render(scene, camera); return; }

        if (isSpeaking) t_anim += delta;

        if (avatar.neck) {
            let neckY = Math.sin(time * 0.38) * 0.022;
            if (avatar.rest.neck) neckY += Math.sin(t_anim * 1.7) * 0.04 * smoothIntensity;
            avatar.neck.rotation.y = (avatar.rest.neck ? avatar.rest.neck.y : 0) + neckY;
            avatar.neck.rotation.z = (avatar.rest.neck ? avatar.rest.neck.z : 0) + Math.sin(time * 0.27) * 0.009;
        }
        if (avatar.spine1) avatar.spine1.rotation.x = 0.02 + Math.sin(time * 0.55) * 0.007;

        const blinkTarget = Math.sin(time * 3.6) > 0.965 ? 1 : 0;
        for (const bm of avatar.blinkMeshes) {
            const inf = bm.mesh.morphTargetInfluences;
            inf[bm.blinkL] = THREE.MathUtils.lerp(inf[bm.blinkL], blinkTarget, 0.42);
            if (bm.blinkR !== null) inf[bm.blinkR] = inf[bm.blinkL];
        }

        gazeTimer -= delta;
        if (gazeTimer <= 0) {
            gazeTargetX = (Math.random() - 0.5) * 0.28;
            gazeTargetY = (Math.random() - 0.5) * 0.14;
            gazeTimer = 2 + Math.random() * 2;
        }
        gazeDirX = THREE.MathUtils.lerp(gazeDirX, gazeTargetX, delta * 1.5);
        gazeDirY = THREE.MathUtils.lerp(gazeDirY, gazeTargetY, delta * 1.5);

        for (const em of avatar.eyeLookMeshes) {
            const inf = em.mesh.morphTargetInfluences;
            const x = gazeDirX, y = gazeDirY;
            for (let i = 0; i < inf.length; i++) inf[i] *= 0.85;
            const clamp = (v) => Math.max(0, Math.min(0.4, v));
            if (x > 0 && em.inL !== null) inf[em.inL] = clamp(x);
            if (x < 0 && em.outL !== null) inf[em.outL] = clamp(-x);
            if (x > 0 && em.outR !== null) inf[em.outR] = clamp(x);
            if (x < 0 && em.inR !== null) inf[em.inR] = clamp(-x);
            if (y > 0 && em.upL !== null) inf[em.upL] = clamp(y);
            if (y < 0 && em.downL !== null) inf[em.downL] = clamp(-y);
            if (y > 0 && em.upR !== null) inf[em.upR] = clamp(y);
            if (y < 0 && em.downR !== null) inf[em.downR] = clamp(-y);
        }

        if (analyzer && isSpeaking) {
            analyzer.getByteFrequencyData(audioData);
            analyzer.getByteTimeDomainData(timeData);
            let sum = 0;
            for (let i = 0; i < audioData.length; i++) { const v = audioData[i] / 255; sum += v * v; }
            const rms = Math.sqrt(sum / audioData.length);
            const noiseGate = 0.035;
            const gated = Math.max(0, rms - noiseGate);
            const rawIntensity = Math.min(gated * 3.4, 1.0);
            smoothIntensity = THREE.MathUtils.lerp(smoothIntensity, rawIntensity, 0.26);

            const vSub = bandMean(1, 10), vLow = bandMean(2, 22), vMid = bandMean(12, 48), vHigh = bandMean(36, 96), vAir = bandMean(72, audioData.length);
            const specSum = vLow + vMid + vHigh + 1e-6;
            const lowN = vLow / specSum, midN = vMid / specSum, highN = vHigh / specSum;

            let centroidNum = 0, centroidDen = 0;
            for (let i = 2; i < audioData.length; i++) { const e = audioData[i] / 255; centroidNum += i * e; centroidDen += e; }
            const centroid = centroidDen > 1e-5 ? centroidNum / centroidDen : 0;
            const cNorm = THREE.MathUtils.clamp(centroid / Math.max(32, audioData.length * 0.85), 0, 1);

            let tdEnergy = 0, zeroCrossings = 0, prevSample = (timeData[0] - 128) / 128;
            for (let i = 0; i < timeData.length; i++) {
                const sample = (timeData[i] - 128) / 128;
                tdEnergy += sample * sample;
                if ((sample >= 0 && prevSample < 0) || (sample < 0 && prevSample >= 0)) zeroCrossings++;
                prevSample = sample;
            }
            tdEnergy = Math.sqrt(tdEnergy / timeData.length);
            const transient = Math.max(0, tdEnergy - prevEnergy);
            prevEnergy = THREE.MathUtils.lerp(prevEnergy, tdEnergy, 0.5);
            const zcrN = THREE.MathUtils.clamp(zeroCrossings / (timeData.length * 0.12), 0, 1);
            const frication = THREE.MathUtils.clamp(highN * 0.9 + zcrN * 0.55 + vAir * 0.35, 0, 1);
            const ampMod = Math.abs(rawIntensity - prevRmsAmp);
            prevRmsAmp = THREE.MathUtils.lerp(prevRmsAmp, rawIntensity, 0.35);

            const gate = THREE.MathUtils.clamp(smoothIntensity * 1.22, 0, 1);
            const vowelWeights = normalizeCompetitive({
                aa: vLow * 1.15 + vMid * 0.55 + (1 - cNorm) * 0.35,
                O: vLow * 1.25 + vSub * 0.4 + midN * 0.2,
                E: midN * 1.05 + highN * 0.95 + cNorm * 0.45,
                I: highN * 1.2 + midN * 0.35 + cNorm * 0.5,
                U: vLow * 1.05 + (1 - midN) * 0.35 + highN * 0.12,
            });

            let tAA = THREE.MathUtils.clamp(vowelWeights.aa * gate * 1.42, 0, 1);
            let tO = THREE.MathUtils.clamp(vowelWeights.O * gate * 1.28, 0, 1);
            let tE = THREE.MathUtils.clamp(vowelWeights.E * gate * 1.28, 0, 1);
            let tI = THREE.MathUtils.clamp(vowelWeights.I * gate * 1.18, 0, 1);
            let tU = THREE.MathUtils.clamp(vowelWeights.U * gate * 1.12, 0, 1);

            const tPP = THREE.MathUtils.clamp(transient * 6.2 * gate * (0.55 + lowN), 0, 1);
            const tDD = THREE.MathUtils.clamp(transient * 5.4 * gate * (0.45 + midN * 1.1), 0, 1);
            const tKK = THREE.MathUtils.clamp(transient * 4.8 * gate * (0.5 + lowN * 0.95), 0, 1);
            const tSS = THREE.MathUtils.clamp(frication * zcrN * 1.15 * gate * (0.55 + highN), 0, 1);
            const tFF = THREE.MathUtils.clamp(frication * gate * (highN * 0.45 + midN * 0.5 + vAir * 0.25), 0, 1);
            const tTH = THREE.MathUtils.clamp(frication * gate * (midN * 0.65 + highN * 0.35) * (1 - transient * 1.8), 0, 1);
            const tCH = THREE.MathUtils.clamp((frication * 0.5 + vAir * 0.45) * gate * (1 - transient * 1.2), 0, 1);
            const tNN = THREE.MathUtils.clamp(gate * midN * vLow * 1.1 * (1 - transient * 2.4) * (0.35 + zcrN * 0.25), 0, 1);
            const tRR = THREE.MathUtils.clamp(ampMod * 5.5 * gate * (midN * 0.85 + lowN * 0.35), 0, 1);

            const conPeak = Math.max(tPP, tDD, tKK, tSS, tFF, tTH, tCH, tNN * 0.9, tRR * 0.85);
            const vowelAtten = THREE.MathUtils.clamp(1 - conPeak * 0.82, 0.12, 1);
            tAA *= vowelAtten; tO *= vowelAtten; tE *= vowelAtten; tI *= vowelAtten; tU *= vowelAtten;

            const targets = {
                aa: tAA, O: tO, E: tE, I: tI, U: tU,
                CH: tCH, SS: tSS, FF: tFF, PP: tPP, TH: tTH, DD: tDD, kk: tKK, nn: tNN, RR: tRR,
                sil: Math.max(0, 1 - smoothIntensity * 2.85 - conPeak * 0.15),
            };

            const rates = {
                aa: LIPEASE_VOWEL, O: LIPEASE_VOWEL, E: LIPEASE_VOWEL, I: LIPEASE_VOWEL, U: LIPEASE_VOWEL,
                CH: LIPEASE_FRIC, SS: LIPEASE_FRIC, FF: LIPEASE_FRIC, TH: LIPEASE_FRIC,
                PP: LIPEASE_PLOSIVE, DD: LIPEASE_PLOSIVE, kk: LIPEASE_PLOSIVE,
                nn: LIPEASE_VOWEL * 0.9, RR: LIPEASE_VOWEL * 0.88, sil: 0.22,
            };
            smoothTowardVis(targets, rates);
            paintVisemeMeshes(0.4);

            const vPeak = Math.max(smoothVis.aa, smoothVis.O, smoothVis.E, smoothVis.I, smoothVis.U);
            const conOpen = Math.max(smoothVis.CH, smoothVis.TH, smoothVis.SS, smoothVis.FF) * 0.38 + Math.max(smoothVis.PP, smoothVis.DD, smoothVis.kk) * 0.22;
            const visemeContrast = Math.abs(smoothVis.aa - smoothVis.O) + Math.abs(smoothVis.aa - smoothVis.E) + Math.abs(smoothVis.O - smoothVis.E) + Math.abs(smoothVis.I - smoothVis.E);
            const jawBase = smoothIntensity * 0.045;
            const jawDynamic = vPeak * 0.19 + visemeContrast * 0.1 + smoothVis.aa * 0.11 + conOpen;
            const targetJaw = THREE.MathUtils.clamp(jawBase + jawDynamic, 0, 0.42);

            for (const jm of avatar.jawMeshes) {
                const inf = jm.mesh.morphTargetInfluences;
                inf[jm.jawOpen] = THREE.MathUtils.lerp(inf[jm.jawOpen], targetJaw, 0.28);
                if (jm.mouthOpen !== null) { inf[jm.mouthOpen] = THREE.MathUtils.lerp(inf[jm.mouthOpen], targetJaw * 0.34, 0.24); }
            }
        } else if (!isSpeaking && (avatar.visemeMeshes.length || avatar.jawMeshes.length)) {
            smoothIntensity = THREE.MathUtils.lerp(smoothIntensity, 0, LIP_DECAY);
            for (const k of LIP_ORDER) { const dest = k === 'sil' ? 1 : 0; smoothVis[k] = THREE.MathUtils.lerp(smoothVis[k], dest, LIP_DECAY); }
            paintVisemeMeshes(LIP_DECAY + 0.04);
            for (const jm of avatar.jawMeshes) {
                const inf = jm.mesh.morphTargetInfluences;
                inf[jm.jawOpen] = THREE.MathUtils.lerp(inf[jm.jawOpen], 0, LIP_DECAY);
                if (jm.mouthOpen !== null) inf[jm.mouthOpen] = THREE.MathUtils.lerp(inf[jm.mouthOpen], 0, LIP_DECAY);
            }
        }

        const energy = smoothIntensity;
        const targetArmWeight = energy > 0.05 ? 1.0 : 0.0;
        armWeight += (targetArmWeight - armWeight) * delta * 1.5;

        if (avatar.spine2 && avatar.rest.spine2) {
            avatar.spine2.rotation.y = avatar.rest.spine2.y + Math.sin(t_anim * 1.2) * 0.05 * energy;
        }

        if (isSpeaking) {
            bodyAnimT += delta;
            if (avatarKeyframes && avatarKeyframes.length >= 2) {
                const kfs = avatarKeyframes;
                const dur = kfs[kfs.length - 1].time;
                if (dur > 0) {
                    let at = ((bodyAnimT % dur) + dur) % dur;
                    let i1 = 0, i2 = 1, found = false;
                    for (let i = 0; i < kfs.length - 1; i++) { if (at >= kfs[i].time && at < kfs[i + 1].time) { i1 = i; i2 = i + 1; found = true; break; } }
                    if (!found) { i1 = kfs.length - 1; i2 = 0; }
                    const i0 = (i1 - 1 + kfs.length) % kfs.length;
                    const i3 = (i2 + 1) % kfs.length;
                    const k0 = kfs[i0], k1 = kfs[i1], k2 = kfs[i2], k3 = kfs[i3];
                    let start = k1.time;
                    let end = (i2 > i1) ? k2.time : dur + k2.time;
                    if (i2 <= i1 && at < start) at += dur;
                    let lt = Math.max(0, Math.min(1, (at - start) / (end - start)));
                    lt = lt * lt * lt * (lt * (lt * 6 - 15) + 10);
                    const cr = (b, ax) => {
                        const p0 = k0.bones[b]?.[ax] ?? 0, p1 = k1.bones[b]?.[ax] ?? 0, p2 = k2.bones[b]?.[ax] ?? 0, p3 = k3.bones[b]?.[ax] ?? 0, t2 = lt * lt, t3 = t2 * lt;
                        return 0.5 * ((2 * p1) + (-p0 + p2) * lt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
                    };
                    const applyBone = (bone, name) => {
                        if (!bone || !k1.bones[name]) return;
                        bone.rotation.x = cr(name, 'x'); bone.rotation.y = cr(name, 'y'); bone.rotation.z = cr(name, 'z');
                    };
                    applyBone(avatar.leftArm, 'LeftArm'); applyBone(avatar.rightArm, 'RightArm');
                    applyBone(avatar.leftForeArm, 'LeftForeArm'); applyBone(avatar.rightForeArm, 'RightForeArm');
                    applyBone(avatar.leftShoulder, 'LeftShoulder'); applyBone(avatar.rightShoulder, 'RightShoulder');
                }
            }
        } else {
            bodyAnimT = 0;
        }

        if(renderer && scene && camera) renderer.render(scene, camera);
    }

    window._destroyAvatar3D = function() {
        if (animFrameId) cancelAnimationFrame(animFrameId);
        containerReady = false;
        avatar.loaded = false;
        avatar.model = null;
        if (analyzer) { try { analyzer.disconnect(); } catch(e){} analyzer = null; }
        if (audioCtx) { audioCtx.close(); audioCtx = null; }
        const container = document.getElementById('avatar3dContainer');
        if (container && renderer) {
            try { container.removeChild(renderer.domElement); } catch(e) {}
            renderer.dispose();
            renderer = null;
        }
    };

    window._playAvatarLipsync = function(url) {
        return new Promise(async (resolve) => {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === 'suspended') await audioCtx.resume();
            
            const href = url.startsWith('http') ? url : (window.location.origin + url);
            const audio = new Audio(`${href}${href.includes('?') ? '&' : '?'}t=${Date.now()}`);
            audio.crossOrigin = 'anonymous';
            const source = audioCtx.createMediaElementSource(audio);
            analyzer = audioCtx.createAnalyser();
            analyzer.fftSize = LIP_FFT;
            analyzer.smoothingTimeConstant = 0.55;
            source.connect(analyzer);
            analyzer.connect(audioCtx.destination);
            
            resetLipState();
            isSpeaking = true;
            
            const cleanup = () => {
                isSpeaking = false;
                resetLipState();
                try { if (analyzer) analyzer.disconnect(); source.disconnect(); } catch (_) { }
                analyzer = null;
            };
            
            audio.addEventListener('ended', () => { cleanup(); resolve(); });
            audio.addEventListener('error', () => { cleanup(); resolve(); });
            audio.play().catch(() => { cleanup(); resolve(); });
        });
    };
</script>
"""

import sys
sim_path = r'c:\Users\MSI\Downloads\Projet DS\alia_django\templates\simulator\index.html'
with open(sim_path, 'r', encoding='utf-8') as f:
    sim_html = f.read()

# Replace the giant <script type="module"> we accidentally pasted earlier
sim_new = re.sub(r'<script type="module">.*?</script>', avatar_script, sim_html, count=1, flags=re.DOTALL)

with open(sim_path, 'w', encoding='utf-8') as f:
    f.write(sim_new)

print("Replaced script successfully.")
