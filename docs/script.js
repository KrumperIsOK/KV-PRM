const copyButton = document.querySelector("#copy-citation");
const citation = document.querySelector("#bibtex");
const status = document.querySelector("#copy-status");

if (navigator.clipboard && copyButton && citation && status) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent);
      status.textContent = "Citation copied.";
    } catch {
      status.textContent =
        "Clipboard access is unavailable. Select the citation text or download the BibTeX file.";
    }
  });
}
