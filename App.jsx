import React, { useState } from 'react';

function App() {
  const [selectedFiles, setSelectedFiles] = useState(null);
  const [status, setStatus] = useState("");
  const [modelUrl, setModelUrl] = useState(null);

  // 1. ファイルが選択されたら、それを記憶する
  const handleFileChange = (event) => {
    setSelectedFiles(event.target.files);
  };

  // 2. 「アップロード」ボタンが押された時の処理
  const handleUpload = async () => {
    if (!selectedFiles) {
      alert("ファイルを選択してください！");
      return;
    }

    setStatus("送信中...");

    // 送信用のデータを作る
    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append("files", selectedFiles[i]);
    }

    try {
      // バックエンド(FastAPI)に送信！
      // ※ ポート番号 8000 はバックエンドの設定に合わせてください
      const response = await fetch("http://127.0.0.1:8000/upload/", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setStatus("成功: " + data.message);
        
        // 生成が完了したらモデルを取得する処理をここに書く（今回は省略）
        // 実際はここからポーリング（定期確認）などが始まります
      } else {
        setStatus("エラーが発生しました。");
      }
    } catch (error) {
      console.error("Error:", error);
      setStatus("サーバーに接続できませんでした。バックエンドは起動していますか？");
    }
  };

  return (
    <div style={{ padding: "50px", textAlign: "center" }}>
      <h1>📸 3Dモデル生成アプリ</h1>
      
      <div style={{ margin: "20px", padding: "20px", border: "2px dashed #ccc" }}>
        <input 
          type="file" 
          multiple 
          onChange={handleFileChange} 
          accept="image/*"
        />
        <p>ここに写真をドラッグ＆ドロップ、または選択</p>
      </div>

      <button 
        onClick={handleUpload} 
        style={{ padding: "10px 20px", fontSize: "16px", cursor: "pointer" }}
      >
        写真をアップロードして生成開始
      </button>

      <p style={{ marginTop: "20px", fontWeight: "bold" }}>{status}</p>
    </div>
  );
}

export default App;
