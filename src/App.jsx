import React, { useState, Suspense, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, OrbitControls, Stage, Html } from '@react-three/drei';
import axios from 'axios';

// 3Dモデルを表示・色味を変えるコンポーネント
function Model({ url, customColor, metalness }) {
  const { scene } = useGLTF(url);

  // 色味と金属感をリアルタイムに変更
  useEffect(() => {
    if (scene) {
      scene.traverse((child) => {
        if (child.isMesh && child.material) {
          // 色を設定
          child.material.color.set(customColor);
          // 金属感を設定
          child.material.metalness = metalness;
          // 反射（粗さ）を少し調整すると金属感がわかりやすくなります
          child.material.roughness = 0.3; 
        }
      });
    }
  }, [scene, customColor, metalness]);

  return <primitive object={scene} />;
}

export default function App() {
  const [image, setImage] = useState(null);
  const [modelUrl, setModelUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // テクスチャ操作用のステート
  const [color, setColor] = useState('#ffffff');
  const [metalness, setMetalness] = useState(0.5);

  // 【ユーザー様提供】URLを受け取るための関数
  const uploadAndGenerate = async () => {
    if (!image) return alert("画像を選んでください");
    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post(
        'https://unstrategical-rubi-nonliterary.ngrok-free.dev/generate', 
        formData,
        {
          headers: {
            'ngrok-skip-browser-warning': 'true'
          }
          // URL文字列を受け取るので responseType: 'blob' は不要
        }
      );

      console.log("受信したデータ:", response.data);

      if (response.data && response.data.glb_url) {
        // バックエンドから届いたURL（アヒルなど）をセット
        setModelUrl(response.data.glb_url); 
      } else {
        alert("サーバーから正しいURLが届きませんでした");
      }

    } catch (error) {
      console.error("通信エラー:", error);
      alert("バックエンドとの通信に失敗しました");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#111', color: 'white', display: 'flex', flexDirection: 'column' }}>
      
      {/* 操作パネル */}
      <div style={{ padding: '20px', background: '#222', zIndex: 10 }}>
        <input type="file" onChange={(e) => setImage(e.target.files[0])} />
        <button onClick={uploadAndGenerate} disabled={loading} style={{ marginLeft: '10px', padding: '5px 15px' }}>
          {loading ? '生成中...' : '3D化する'}
        </button>

        {/* モデルが表示されたら操作UIを出す */}
        {modelUrl && (
          <div style={{ marginTop: '15px', display: 'flex', gap: '20px', alignItems: 'center' }}>
            <label>
              色味: <input type="color" value={color} onChange={(e) => setColor(e.target.value)} />
            </label>
            <label>
              金属感: 
              <input type="range" min="0" max="1" step="0.1" value={metalness} onChange={(e) => setMetalness(parseFloat(e.target.value))} />
              {metalness}
            </label>
          </div>
        )}
      </div>

      {/* 3D表示エリア */}
      <div style={{ flex: 1, position: 'relative' }}>
        <Canvas shadows camera={{ position: [0, 0, 3] }}>
          <Suspense fallback={<Html center><div style={{color: 'white'}}>読み込み中...</div></Html>}>
            <Stage environment="city" intensity={0.5}>
              {modelUrl && <Model url={modelUrl} customColor={color} metalness={metalness} />}
            </Stage>
          </Suspense>
          <OrbitControls makeDefault />
        </Canvas>
      </div>
    </div>
  );
}