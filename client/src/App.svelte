<script lang="ts">
  import { onMount, tick } from "svelte";
  import SearchResults from "./components/SearchResults.svelte";
  import TextView from "./components/TextView.svelte";
  import Tailwindcss from "./Tailwind.svelte";
  import SearchBar from "./components/SearchBar.svelte";
  import {
    type File,
    type Navigation,
    type PdfPosition,
    type SearchResultSet,
    type Preference,
    preferenceKey,
    type ParsedQuery,
  } from "./types";
  import PdfView from "./components/PdfView.svelte";
  import TabBar from "./components/TabBar.svelte";

  // base URL for the Semantra API
  const API_BASE_URL = ""; // Default semantra port is usually 5000

  let files: File[] = [];
  let activeFileIndex = 0;
  let tokens: string[] = [];
  let text: string | null = null;
  let pdfPositions: PdfPosition[] = [];
  let updating = false;
  let unsearched = true;
  let searchResultsElem: SearchResults;
  let currentSearchTerm = "";
  let uploading = false;
  let uploadError: string | null = null;
  let uploadedFilesExpanded = false; // collapse the uploaded files list by default

  let preferences: { [key: string]: Preference } = {};
  $: activeFile =
    activeFileIndex < files.length ? files[activeFileIndex] : null;
  $: filesByPath = Object.fromEntries(
    files.map((file) => [file.filename, file]),
  );
  $: fileIndicesByPath = Object.fromEntries(
    files.map((file, index) => [file.filename, index]),
  );

  $: updateFile(activeFile);

  async function updateFile(file: File | null) {
    // Reset everything
    tokens = [];
    text = null;
    pdfPositions = [];

    if (file == null) return;

    updating = true;
    uploadError = null;

    try {
      // Get text
      const textResponse = await fetch(
        `${API_BASE_URL}/api/text?filename=${encodeURIComponent(file.filename)}`,
      );

      if (!textResponse.ok) {
        throw new Error(`Failed to load text: ${textResponse.statusText}`);
      }

      tokens = await textResponse.json();
      text = tokens.join("");

      if (file.filetype === "pdf") {
        const pdfResponse = await fetch(
          `${API_BASE_URL}/api/pdfpositions?filename=${encodeURIComponent(file.filename)}`,
        );

        if (!pdfResponse.ok) {
          throw new Error(
            `Failed to load PDF positions: ${pdfResponse.statusText}`,
          );
        }

        pdfPositions = await pdfResponse.json();
      }

      await tick();
      navigate();
    } catch (error) {
      console.error("Error loading file:", error);
      uploadError = `Error loading file: ${error instanceof Error ? error.message : "Unknown error"}`;
    } finally {
      updating = false;
    }
  }

  let searchResultSet: SearchResultSet = {
    results: [],
    sort: "asc",
  };

  let textView: TextView;
  let pdfView: PdfView;
  let searchBar: SearchBar;

  export function parseQuery(query: string): ParsedQuery[] {
    // Parse the query
    // e.g. "dog + cat" => [{query: "dog", weight: 1}, {query: "cat", weight: 1}]
    // e.g. "dog - cat" => [{query: "dog", weight: 1}, {query: "cat", weight: -1}]
    // e.g. "dog is cool - cat" => [{query: "dog is cool", weight: 1}, {query: "cat", weight: -1}]
    // e.g. "dog +1.2 cat" => [{query: "dog", weight: 1}, {query: "cat", weight: 1.2}]
    // e.g. "+3 dogs are nice -2 cats are mean" => [{query: "dogs are nice", weight: 3}, {query: "cats are mean", weight: 2}]
    // Parse the query
    const regex = /([\+\-]?\d*\.?\d*\s*)?([^\+\-]+)/g;
    const parsedQueries: ParsedQuery[] = [];

    let match;
    while ((match = regex.exec(query)) !== null) {
      const weight =
        parseFloat(match[1]) || (match[1] && match[1].includes("-") ? -1 : 1);
      const searchTerm = match[2].trim();
      parsedQueries.push({ query: searchTerm, weight });
    }

    return parsedQueries;
  }

  function scrollSearchResultsToTop() {
    if (searchResultsElem) searchResultsElem.scrollToTop();
  }

  async function handleSearch(query: string) {
    currentSearchTerm = query;
    const preferenceValues = Object.values(preferences)
      .filter((preference) => preference.weight !== 0)
      .map((x) => ({ ...x }));

    // Ignore empty queries
    if (query.trim() === "" && preferenceValues.length === 0) {
      searchResultSet = {
        results: [],
        sort: "asc",
      };
      scrollSearchResultsToTop();
      return;
    }
    const parsedQueries = parseQuery(query);

    // Adjust weights so that all positive weights are split evenly
    // and all negative weights are split evenly, and the sum of all
    // weights is 1
    const POSITIVE_RATIO = 0.61803398875;
    const NEGATIVE_RATIO = 1 - POSITIVE_RATIO;

    const totalPositiveCount =
      parsedQueries.filter((query) => query.weight > 0).length +
        preferenceValues.filter((preference) => preference.weight > 0).length ||
      1;
    const totalNegativeCount =
      parsedQueries.filter((query) => query.weight < 0).length +
        preferenceValues.filter((preference) => preference.weight < 0).length ||
      1;
    for (const query of parsedQueries) {
      if (query.weight > 0) {
        query.weight *= POSITIVE_RATIO / totalPositiveCount;
      } else if (query.weight < 0) {
        query.weight *= NEGATIVE_RATIO / totalNegativeCount;
      }
    }
    for (const preference of preferenceValues) {
      if (preference.weight > 0) {
        preference.weight *= POSITIVE_RATIO / totalPositiveCount;
      } else if (preference.weight < 0) {
        preference.weight *= NEGATIVE_RATIO / totalNegativeCount;
      }
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          queries: parsedQueries,
          preferences: preferenceValues,
        }),
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      searchResultSet = await response.json();
      sidebarExpanded = true;
      scrollSearchResultsToTop();
      unsearched = false;
    } catch (error) {
      console.error("Search error:", error);
      uploadError = `Search failed: ${error.message}`;
    }
  }

  onMount(async () => {
    try {
      const filesResponse = await fetch(`${API_BASE_URL}/api/files`);
      if (filesResponse.ok) {
        files = await filesResponse.json();
      } else {
        console.error("Error fetching files:", filesResponse.statusText);
      }
    } catch (error) {
      console.error("Error connecting to server:", error);
    }
  });

  $: tokenOffsets = tokens.reduce(
    (acc, token) => {
      const lastOffset = acc[acc.length - 1];
      acc.push(lastOffset + token.length);
      return acc;
    },
    [0],
  );

  let pendingNavigation: Navigation | null = null;

  async function navigate() {
    if (pendingNavigation == null) return;
    //sidebarExpanded = false;
    if (textView) {
      textView.navigate(
        tokenOffsets[pendingNavigation.searchResult.offset[0]],
        tokenOffsets[pendingNavigation.searchResult.offset[1]],
      );
    } else if (pdfView) {
      pdfView.navigate(
        tokenOffsets[pendingNavigation.searchResult.offset[0]],
        tokenOffsets[pendingNavigation.searchResult.offset[1]],
      );
    }
    pendingNavigation = null;

    await tick();
    // Scroll active tab into view
    const activeTab = document.querySelector(".active-tab");
    if (activeTab) {
      activeTab.scrollIntoView({
        inline: "center",
      });
    }
  }

  async function jumpToResult(result: Navigation) {
    pendingNavigation = result;
    const newFileIndex = fileIndicesByPath[result.file.filename];
    if (newFileIndex !== activeFileIndex) {
      activeFileIndex = newFileIndex;
    } else {
      navigate();
    }
  }

  function focusFile(index: number) {
    if (index < 0 || index >= files.length) return;
    activeFileIndex = index;
  }

  function setPreference(preference: Preference) {
    preferences[preferenceKey(preference.file, preference.searchResult)] =
      preference;
    if (searchBar != null) searchBar.scrollToBottomOfPreferences();
  }

  async function handleFileUpload(event) {
    const selectedFiles = Array.from(event.target.files as FileList);
    if (selectedFiles.length === 0) {
      uploadError = "Please select at least one file";
      return;
    }

    uploading = true;
    uploadError = null;

    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      console.log(
        `Uploading ${selectedFiles.length} files to ${API_BASE_URL}/api/upload`,
      );

      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Upload failed (${response.status}): ${errorText || response.statusText}`,
        );
      }

      const uploadResult = await response.json();
      const processedFiles = uploadResult.processed_files ?? [];
      const failedFiles = processedFiles.filter(
        (file) => file.status !== "success",
      );
      const succeededFiles = processedFiles.filter(
        (file) => file.status === "success",
      );

      if (failedFiles.length > 0) {
        uploadError = failedFiles
          .map((file) =>
            file.error
              ? `${file.basename}: ${file.error}`
              : `${file.basename}: failed`,
          )
          .join("; ");
      }

      if (succeededFiles.length > 0) {
        // Refresh the file list
        const filesResponse = await fetch(`${API_BASE_URL}/api/files`);
        if (filesResponse.ok) {
          const previousPaths = new Set(files.map((file) => file.filename));
          const updatedFiles: File[] = await filesResponse.json();
          files = updatedFiles;
          const firstNewIndex = updatedFiles.findIndex(
            (file) => !previousPaths.has(file.filename),
          );
          if (firstNewIndex !== -1) {
            activeFileIndex = firstNewIndex;
          }
        } else {
          console.error(
            "Failed to refresh file list:",
            filesResponse.statusText,
          );
        }
      }

      // Reset file input
      event.target.value = "";
    } catch (error) {
      console.error("Upload error:", error);
      uploadError = error.message;
    } finally {
      uploading = false;
    }
  }

  async function deleteDocument(filename) {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) {
      return;
    }

    uploadError = null;

    try {
      const response = await fetch(`${API_BASE_URL}/api/delete`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ filename }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Delete failed: ${errorText || response.statusText}`);
      }

      // Update the files list by removing the deleted file
      files = files.filter((file) => file.filename !== filename);

      // Also update the search results to remove references to the deleted file
      searchResultSet = {
        ...searchResultSet,
        results: searchResultSet.results.filter(
          ([resultFilename]) => resultFilename !== filename,
        ),
      };

      // Clear preferences related to the deleted file
      for (const key in preferences) {
        if (preferences[key].file.filename === filename) {
          delete preferences[key];
        }
      }

      // If the deleted file was the active file, select another one if available
      if (activeFileIndex >= files.length && files.length > 0) {
        activeFileIndex = files.length - 1;
      }
    } catch (error) {
      console.error("Error deleting document:", error);
      uploadError = `Failed to delete document: ${error.message}`;
    }
  }

  let sidebarExpanded = true;
</script>

<Tailwindcss />

<main class="flex flex-col h-full bg-slate-100">
  <header
    class="flex flex-row border-b-4 border-black py-4 px-8 max-lg:px-4 items-start"
  >
    <h1 class="text-3xl font-mono font-bold inline-flex pr-6 mt-1">Semantra</h1>
    <div class="flex-1 flex items-center">
      <SearchBar
        bind:this={searchBar}
        {preferences}
        on:setPreference={(e) => setPreference(e.detail)}
        on:search={(e) => handleSearch(e.detail)}
      />
    </div>
    <div class="flex items-center ml-4">
      <div class="relative">
        <input
          type="file"
          id="file-upload"
          multiple
          accept=".pdf,.txt,.html,.xlsx,.xlsm,.xltx,.xltm"
          on:change={handleFileUpload}
          disabled={uploading}
          class="absolute inset-0 opacity-0 w-full h-full cursor-pointer"
        />
        <button
          class="bg-gray-400 hover:bg-gray-500 text-white font-medium py-2 px-4 rounded flex items-center"
          class:opacity-50={uploading}
          class:pointer-events-none={uploading}
        >
          {#if uploading}
            <svg
              class="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                class="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="4"
              ></circle>
              <path
                class="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
            Uploading...
          {:else}
            Upload Files
          {/if}
        </button>
      </div>
    </div>
  </header>

  {#if files.length > 0}
    <section class="bg-white border-b border-slate-200 px-8 py-3 max-lg:px-4">
      <button
        type="button"
        class="flex w-full items-center justify-between mb-2 text-xs uppercase tracking-wide text-slate-500"
        on:click={() => (uploadedFilesExpanded = !uploadedFilesExpanded)}
        aria-expanded={uploadedFilesExpanded}
        aria-controls="uploaded-files-list"
      >
        <span>已上传文件</span>
        <span class="flex items-center gap-1">
          <span>{files.length}</span>
          <svg
            class={`h-3 w-3 transition-transform duration-200 ${uploadedFilesExpanded ? "rotate-180" : ""}`}
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            stroke-width="2"
          >
            <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </span>
      </button>
      <div
        id="uploaded-files-list"
        class="max-h-36 overflow-y-auto pr-1"
        class:hidden={!uploadedFilesExpanded}
      >
        <ul class="flex flex-wrap gap-2">
          {#each files as file, index}
            <li>
              <button
                type="button"
                class="px-3 py-2 border border-slate-300 rounded text-sm bg-slate-50 hover:bg-slate-100 transition-colors text-left"
                class:bg-black={index === activeFileIndex}
                class:text-white={index === activeFileIndex}
                class:border-black={index === activeFileIndex}
                on:click={() => focusFile(index)}
              >
                <div class="font-medium">{file.basename}</div>
                <div class="text-xs opacity-70 uppercase tracking-wide">
                  {file.filetype}
                </div>
              </button>
            </li>
          {/each}
        </ul>
      </div>
    </section>
  {/if}

  {#if uploadError}
    <div
      class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 m-2"
      role="alert"
    >
      <p class="font-bold">Error</p>
      <p>{uploadError}</p>
    </div>
  {/if}

  <article class="flex flex-1 flex-row relative items-stretch">
    <SearchResults
      bind:sidebarExpanded
      bind:this={searchResultsElem}
      {unsearched}
      {preferences}
      on:setPreference={(e) => setPreference(e.detail)}
      on:navigate={(e) => jumpToResult(e.detail)}
      {activeFile}
      {filesByPath}
      {searchResultSet}
    />
    <div class="flex flex-col flex-1">
      {#if activeFile != null}
        <TabBar
          disabled={updating}
          bind:index={activeFileIndex}
          {files}
          onDelete={(filename) => deleteDocument(filename)}
        />
        {#if activeFile.filetype === "pdf"}
          <PdfView
            bind:this={pdfView}
            file={activeFile}
            positions={pdfPositions}
          />
        {:else}
          <TextView
            bind:this={textView}
            text={text == null ? "Loading..." : text}
          />
        {/if}
      {:else if files.length === 0}
        <div class="flex flex-col items-center justify-center h-full">
          <p class="text-xl font-semibold mb-4">
            No files available for search
          </p>
          <p class="text-gray-600 mb-6">
            Upload files using the button in the top right
          </p>
        </div>
      {:else}
        <div class="text-gray-600 ml-2 mt-2 text-sm">Loading...</div>
      {/if}
    </div>
  </article>
  <footer class="bg-black text-white py-1 px-4 text-sm">
    <a
      class="underline mr-4"
      href="https://github.com/freedmand/semantra/blob/main/docs/help.md"
      target="_blank">Help</a
    >
    <a
      class="underline mr-4"
      href="https://github.com/freedmand/semantra/blob/main/docs/tutorial.md"
      target="_blank">Tutorial</a
    >
    <a
      class="underline"
      href="https://github.com/freedmand/semantra"
      target="_blank">Source code</a
    >
  </footer>
</main>

<style>
  :global(html, body) {
    @apply h-full;
  }
</style>
