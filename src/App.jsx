import React, { useState, Suspense, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, OrbitControls, Stage, Html, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';
import axios from 'axios';

// 3Dモデルを表示・設定を反映するコンポーネント
function Model({ url, settings, mode }) {
  const { scene } = useGLTF(url);

  useEffect(() => {
    if (scene) {
      scene.traverse((child) => {
        if (child.isMesh && child.material) {
          if (mode === 'original') {
            child.material.color.set('#ffffff'); 
            child.material.metalness = 0.2;
            child.material.roughness = 0.8;
            child.material.transmission = 0;
          } else {
            child.material.color.set(settings.color);
            child.material.metalness = settings.metalness;
            child.material.roughness = settings.roughness;
            child.material.transmission = settings.transmission;
            child.material.ior = settings.ior;
            child.material.clearcoat = settings.clearcoat;
          }
          child.material.wireframe = settings.wireframe;
          child.material.transparent = true;
          child.material.opacity = settings.opacity;
          child.material.needsUpdate = true;
        }
      });
    }
  }, [scene, settings, mode]);

  return <primitive object={scene} scale={settings.scale} />;
}

export default function App() {
  const [image, setImage] = useState(null);
  const [modelUrl, setModelUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeMode, setActiveMode] = useState('original');
  const orbitRef = useRef();

  const [settings, setSettings] = useState({
    color: '#ffffff',
    bgColor: '#111111', // ⬅️ 背景色の初期値を追加
    metalness: 0.2,
    roughness: 0.8,
    opacity: 1.0,
    transmission: 0,
    ior: 1.5,
    clearcoat: 0,
    wireframe: false,
    scale: 1.0,
    autoRotate: false,
    environment: 'city',
    intensity: 0.5
  });

  const applyMode = (mode) => {
    setActiveMode(mode);
    const presets = {
      original: { color: '#ffffff', metalness: 0.2, roughness: 0.8, transmission: 0, clearcoat: 0 },
      metal: { color: '#aaaaaa', metalness: 1.0, roughness: 0.1, transmission: 0, clearcoat: 0.5 },
      glass: { color: '#ffffff', metalness: 0, roughness: 0, transmission: 1.0, ior: 1.5, clearcoat: 1.0 },
      neon: { color: '#00ffcc', metalness: 0.5, roughness: 0.2, transmission: 0, clearcoat: 1.0 }
    };
    setSettings(prev => ({ ...prev, ...presets[mode] }));
  };

  const updateSetting = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const uploadAndGenerate = async () => {
    if (!image) return alert("画像を選んでください");
    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);
    try {
      const response = await axios.post(
        'https://unstrategical-rubi-nonliterary.ngrok-free.dev/generate', 
        formData,
        { headers: { 'ngrok-skip-browser-warning': 'true' } }
      );
      if (response.data && response.data.glb_url) {
        setModelUrl(response.data.glb_url);
        applyMode('original');
      }
    } catch (error) {
      alert("通信エラー");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#111', color: 'white', display: 'flex', overflow: 'hidden' }}>
      
      <div style={{ width: '340px', background: '#1a1a1a', padding: '20px', overflowY: 'auto', borderRight: '1px solid #333', zIndex: 10 }}>
        <h2 style={{ fontSize: '1.2rem', marginBottom: '20px' }}>💎 3D Pro Editor</h2>
        
        <section style={{ marginBottom: '25px' }}>
          <input type="file" onChange={(e) => setImage(e.target.files[0])} style={{ width: '100%', fontSize: '12px' }} />
          <button onClick={uploadAndGenerate} disabled={loading} style={mainBtnStyle(loading)}>
            {loading ? '生成中...' : '3Dモデル生成'}
          </button>
        </section>

        {modelUrl && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* 背景設定（新設） */}
            <section style={{ background: '#252525', padding: '15px', borderRadius: '10px' }}>
              <p style={labelStyle}>背景設定</p>
              <label style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                背景の色: <input type="color" value={settings.bgColor} onChange={(e) => updateSetting('bgColor', e.target.value)} />
              </label>
            </section>

            <section>
              <p style={labelStyle}>スタイル選択</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <button onClick={() => applyMode('original')} style={modeBtnStyle(activeMode === 'original')}>基本 (元色)</button>
                <button onClick={() => applyMode('metal')} style={modeBtnStyle(activeMode === 'metal')}>金属</button>
                <button onClick={() => applyMode('glass')} style={modeBtnStyle(activeMode === 'glass')}>ガラス</button>
                <button onClick={() => applyMode('neon')} style={modeBtnStyle(activeMode === 'neon')}>ネオン</button>
              </div>
            </section>

            <section style={{ background: '#252525', padding: '15px', borderRadius: '10px' }}>
              <p style={labelStyle}>質感の微調整</p>
              <div style={{ display: 'grid', gap: '12px', fontSize: '13px' }}>
                {activeMode !== 'original' && (
                  <label>ベース色: <input type="color" value={settings.color} onChange={(e) => updateSetting('color', e.target.value)} /></label>
                )}
                {(activeMode === 'metal' || activeMode === 'original') && (
                  <>
                    <label>金属感: <input type="range" min="0" max="1" step="0.01" value={settings.metalness} onChange={(e) => updateSetting('metalness', parseFloat(e.target.value))} /></label>
                    <label>粗さ: <input type="range" min="0" max="1" step="0.01" value={settings.roughness} onChange={(e) => updateSetting('roughness', parseFloat(e.target.value))} /></label>
                  </>
                )}
                {activeMode === 'glass' && (
                  <>
                    <label>透明度: <input type="range" min="0" max="1" step="0.01" value={settings.transmission} onChange={(e) => updateSetting('transmission', parseFloat(e.target.value))} /></label>
                    <label>屈折率: <input type="range" min="1" max="2.5" step="0.01" value={settings.ior} onChange={(e) => updateSetting('ior', parseFloat(e.target.value))} /></label>
                  </>
                )}
                <label>不透明度: <input type="range" min="0" max="1" step="0.01" value={settings.opacity} onChange={(e) => updateSetting('opacity', parseFloat(e.target.value))} /></label>
                <label><input type="checkbox" checked={settings.wireframe} onChange={(e) => updateSetting('wireframe', e.target.checked)} /> ワイヤーフレーム</label>
              </div>
            </section>

            <section style={{ background: '#252525', padding: '15px', borderRadius: '10px' }}>
              <div style={{ display: 'grid', gap: '10px', fontSize: '13px' }}>
                <label>照明: 
                  <select value={settings.environment} onChange={(e) => updateSetting('environment', e.target.value)} style={{ marginLeft: '10px', background: '#333', color: 'white', border: 'none' }}>
                    <option value="city">City</option>
                    <option value="apartment">Studio</option>
                    <option value="park">Park</option>
                  </select>
                </label>
                <label>大きさ: <input type="range" min="0.5" max="2" step="0.1" value={settings.scale} onChange={(e) => updateSetting('scale', parseFloat(e.target.value))} /></label>
                <label><input type="checkbox" checked={settings.autoRotate} onChange={(e) => updateSetting('autoRotate', e.target.checked)} /> 自動回転</label>
              </div>
            </section>
          </div>
        )}
      </div>

      <div style={{ flex: 1, position: 'relative' }}>
        <Canvas shadows camera={{ position: [0, 0, 4] }}>
          {/* 🌟 背景色を反映するタグを追加 */}
          <color attach="background" args={[settings.bgColor]} />
          
          <Suspense fallback={<Html center><div style={{color: 'white'}}>Loading...</div></Html>}>
            <Stage environment={settings.environment} intensity={settings.intensity} contactShadow={false}>
              {modelUrl && <Model url={modelUrl} settings={settings} mode={activeMode} />}
            </Stage>
            <ContactShadows position={[0, -1, 0]} opacity={0.4} scale={10} blur={2} far={4} />
          </Suspense>
          <OrbitControls ref={orbitRef} makeDefault autoRotate={settings.autoRotate} />
        </Canvas>
      </div>
    </div>
  );
}

const mainBtnStyle = (loading) => ({
  width: '100%', marginTop: '10px', padding: '10px', 
  background: loading ? '#444' : '#007bff', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer'
});

const modeBtnStyle = (active) => ({
  padding: '8px', fontSize: '12px', borderRadius: '4px', border: 'none',
  background: active ? '#007bff' : '#333', color: 'white', cursor: 'pointer', transition: '0.2s'
});

const labelStyle = { fontSize: '12px', color: '#aaa', marginBottom: '8px', fontWeight: 'bold' };