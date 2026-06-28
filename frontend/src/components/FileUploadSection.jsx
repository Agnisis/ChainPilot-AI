export default function FileUploadSection({
  title,
  description,
  accept,
  files,
  onUpload,
  loading,
  buttonLabel,
}) {
  return (
    <section className="panel upload-panel">
      <h2>{title}</h2>
      <p className="muted">{description}</p>
      <label className="upload-drop">
        <input
          type="file"
          accept={accept}
          multiple
          disabled={loading}
          onChange={(e) => {
            const selected = Array.from(e.target.files || []);
            if (selected.length) onUpload(selected);
            e.target.value = "";
          }}
        />
        <span>{buttonLabel || "Choose files or drag here"}</span>
      </label>
    </section>
  );
}
