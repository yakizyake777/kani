import React, { useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, OrbitControls, Stage, Html } from '@react-three/drei';
import axios from 'axios';

// モデル表示コンポーネント（エラーハンドリング付き）
function Model({ url }) {
  try {
    const { scene } = useGLTF(url);
    return <primitive object={scene} />;
  } catch (error) {
    return <Html center><div style={{color: "red"}}>モデル読み込みエラー</div></Html>;
  }
}

export default function App() {
  const [image, setImage] = useState(null);
  const [modelUrl, setModelUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    if (e.target.files[0]) {
      setImage(e.target.files[0]);
      setModelUrl(null);
    }
  };

  const uploadAndGenerate = async () => {
    if (!image) return alert("画像を選んでください");
    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post('https://unstrategical-rubi-nonliterary.ngrok-free.dev/generate', 
      formData,
      {
        headers: {
          'ngrok-skip-browser-warning': 'true' // これを追加
        }
      }
    );
      console.log("受信データ:", response.data); // デバッグ用
      setModelUrl(response.data.glb_url);
    } catch (error) {
      console.error("通信エラー:", error);
      alert("バックエンドとの通信に失敗しました");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column', background: '#111', color: 'white' }}>
      <div style={{ padding: '20px', background: '#222', display: 'flex', gap: '15px', zIndex: 10 }}>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button onClick={uploadAndGenerate} disabled={loading} style={{ padding: '8px 16px', cursor: 'pointer' }}>
          {loading ? '生成中...' : '3D化する'}
        </button>
      </div>

      <div style={{ flex: 1, position: 'relative' }}>
        <Canvas>
          {/* Suspenseで囲むことで真っ白になるのを防ぐ */}
          <Suspense fallback={<Html center><p>読み込み中...</p></Html>}>
            <Stage environment="city" intensity={0.5}>
              {modelUrl && <Model url={modelUrl} />}
            </Stage>
          </Suspense>
          <OrbitControls title="マウスで回転" />
        </Canvas>
      </div>
    </div>
  );
}
