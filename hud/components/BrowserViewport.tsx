export default function BrowserViewport({ screenshotPath }: { screenshotPath: string }) {
  if (!screenshotPath) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-500 text-sm">
        Waiting for agent screenshot...
      </div>
    );
  }
  return (
    <div className="h-full bg-zinc-950 rounded overflow-hidden flex items-center justify-center">
      <img
        src={screenshotPath}
        alt="Agent browser viewport"
        className="max-w-full max-h-full object-contain transition-opacity duration-300"
        onError={(e) => {
          (e.currentTarget as HTMLImageElement).style.display = "none";
        }}
      />
    </div>
  );
}
