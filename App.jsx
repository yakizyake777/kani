import { Canvas } from '@react-three/fiber'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'
import { useLoader } from '@react-three/fiber'

function Model({ url }) {
  // バックエンドから受け取ったURLのモデルを表示
  const geometry = useLoader(PLYLoader, url)
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial vertexColors={true} />
    </mesh>
  )
}

export default function App() {
  // ... ここにアップロードボタンの処理などを書く ...
  
  return (
    <div id="canvas-container">
      <Canvas>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        {/* モデルのURLがあるときだけ表示 */}
        {modelUrl && <Model url={modelUrl} />}
      </Canvas>
    </div>
  )
}
