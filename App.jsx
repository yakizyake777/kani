import React, { useState, Suspense, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, OrbitControls, Stage, Html, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';
import axios from 'axios';

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
const handleImageClick = (e, points, setPoints) => {
    if (points.length >= 2) return; // すでに2点選ばれていたら無視

    // クリックした要素（imgタグ）の情報を取得
    const imgElement = e.target;
    const rect = imgElement.getBoundingClientRect();
    
    // 画面上の表示サイズにおけるクリック位置 (0〜clientWidth)
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;

    // 画像の本来のサイズ (naturalWidth) と表示サイズ (clientWidth) の比率を使って、
    // 「実際の画像の何ピクセル目をクリックしたか」を計算する
    const originalX = Math.round((displayX / imgElement.clientWidth) * imgElement.naturalWidth);
    const originalY = Math.round((displayY / imgElement.clientHeight) * imgElement.naturalHeight);

    // 画面に赤い点を描画するための情報(display)と、Pythonに送るための情報(original)を保存
    setPoints([...points, { displayX, displayY, originalX, originalY }]);
  };
const handleImageChange = (e, setImage, setPreview, setPoints) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file)); // ブラウザ上で表示するためのURLを生成
      setPoints([]); // 画像が変わったらクリックした点もリセット
    }
  };
export default function App() {
  // 🌟 1. 画像ステートを2つに増やす
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);

  const [preview1, setPreview1] = useState(null);
  const [preview2, setPreview2] = useState(null);
  const [points1, setPoints1] = useState([]); // [{x: 120, y: 300}, ...] 実際のピクセル座標
  const [points2, setPoints2] = useState([]);
  const [modelUrl, setModelUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeMode, setActiveMode] = useState('original');
  const orbitRef = useRef();

  const [settings, setSettings] = useState({
    color: '#ffffff',
    bgColor: '#111111',
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
    // 🌟 2. 両方の画像が選ばれているか確認
    if (!image1 || !image2) return alert("画像を2枚とも選んでください");
    
    if (points1.length !== 2 || points2.length !== 2) {
      return alert("画像1、画像2それぞれで上端と下端の「2点」をクリックして指定してください");
    }
    setLoading(true);
    const formData = new FormData();
    // 🌟 3. 両方の画像をFormDataに追加する
    formData.append('image1', image1);
    formData.append('image2', image2);

    const coords1 = points1.map(p => [p.originalX, p.originalY]);
    const coords2 = points2.map(p => [p.originalX, p.originalY]);
    formData.append('points1', JSON.stringify(coords1)); // 例: "[[100, 200], [100, 800]]"
    formData.append('points2', JSON.stringify(coords2));

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
        <h2 style={{ fontSize: '1.2rem', marginBottom: '20px' }}>3D view</h2>
        
<section style={{ marginBottom: '25px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          
          {/* 画像1の入力とプレビュー */}
          <div style={{ background: '#222', padding: '10px', borderRadius: '8px' }}>
            <p style={{fontSize: '12px', color: '#ddd', marginBottom: '5px'}}>画像1 (正面) - 上・下の順に2点クリック</p>
            <input type="file" onChange={(e) => handleImageChange(e, setImage1, setPreview1, setPoints1)} style={{ width: '100%', fontSize: '12px', marginBottom: '10px' }} />
            
            {/* プレビュー画像と赤い点の表示エリア */}
            {preview1 && (
              <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
                {/* プレビュー画像本体 */}
                <img 
                  src={preview1} 
                  alt="preview 1" 
                  onClick={(e) => handleImageClick(e, points1, setPoints1)}
                  style={{ width: '100%', cursor: 'crosshair', borderRadius: '4px', border: '1px solid #444' }} 
                />
                
                {/* クリックした座標に赤い丸を描画 */}
                {points1.map((p, index) => (
                  <div key={index} style={{
                    position: 'absolute', left: p.displayX, top: p.displayY,
                    width: '12px', height: '12px', background: 'red', border: '2px solid white', borderRadius: '50%',
                    transform: 'translate(-50%, -50%)', pointerEvents: 'none'
                  }}>
                    <span style={{ position:'absolute', top:'-15px', left:'15px', color:'white', fontSize:'10px', fontWeight:'bold', textShadow:'1px 1px 2px black' }}>
                      {index + 1}
                    </span>
                  </div>
                ))}
                
                {/* リセットボタン */}
                {points1.length > 0 && (
                  <button onClick={() => setPoints1([])} style={{ fontSize:'11px', marginTop:'8px', padding:'4px 8px', background:'#555', color:'white', border:'none', borderRadius:'4px', cursor:'pointer' }}>
                    点の選択をやり直す
                  </button>
                )}
              </div>
            )}
          </div>

          {/* 画像2の入力とプレビュー */}
          <div style={{ background: '#222', padding: '10px', borderRadius: '8px' }}>
            <p style={{fontSize: '12px', color: '#ddd', marginBottom: '5px'}}>画像2 (側面) - 上・下の順に2点クリック</p>
            <input type="file" onChange={(e) => handleImageChange(e, setImage2, setPreview2, setPoints2)} style={{ width: '100%', fontSize: '12px', marginBottom: '10px' }} />
            
            {/* プレビュー画像と赤い点の表示エリア */}
            {preview2 && (
              <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
                <img 
                  src={preview2} 
                  alt="preview 2" 
                  onClick={(e) => handleImageClick(e, points2, setPoints2)}
                  style={{ width: '100%', cursor: 'crosshair', borderRadius: '4px', border: '1px solid #444' }} 
                />
                
                {points2.map((p, index) => (
                  <div key={index} style={{
                    position: 'absolute', left: p.displayX, top: p.displayY,
                    width: '12px', height: '12px', background: 'red', border: '2px solid white', borderRadius: '50%',
                    transform: 'translate(-50%, -50%)', pointerEvents: 'none'
                  }}>
                    <span style={{ position:'absolute', top:'-15px', left:'15px', color:'white', fontSize:'10px', fontWeight:'bold', textShadow:'1px 1px 2px black' }}>
                      {index + 1}
                    </span>
                  </div>
                ))}
                
                {points2.length > 0 && (
                  <button onClick={() => setPoints2([])} style={{ fontSize:'11px', marginTop:'8px', padding:'4px 8px', background:'#555', color:'white', border:'none', borderRadius:'4px', cursor:'pointer' }}>
                    点の選択をやり直す
                  </button>
                )}
              </div>
            )}
          </div>

          {/* 送信ボタン */}
          <button onClick={uploadAndGenerate} disabled={loading} style={mainBtnStyle(loading)}>
            {loading ? '生成中...' : '3Dモデル生成'}
          </button>
        </section>

        {modelUrl && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
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