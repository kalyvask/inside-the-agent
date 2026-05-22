export default function BrowserViewport({ screenshotPath }: { screenshotPath: string }) {
  if (!screenshotPath) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-500">
        Waiting for agent screenshot...
      </div>
    );
  }
  return (
    <div className="h-full bg-zinc-950 rounded overflow-hidden">
      {/* Next.js Image disabled here for local file paths during demo */}
      <img src={screenshotPath} alt="Agent browser viewport" className="w-full h-full object-contain" />
    </div>
  );
}
