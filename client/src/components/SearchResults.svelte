<script lang="ts">
  import type {
    File,
    Preference,
    SearchResult,
    SearchResultSet,
    ScoredSearchResult,
  } from "../types";
  import SearchResultComponent from "./SearchResult.svelte";

  export let searchResultSet: SearchResultSet;
  export let filesByPath: { [path: string]: File };
  export let preferences: { [key: string]: Preference };
  export let activeFile: File | null;
  export let unsearched: boolean;
  export let sidebarExpanded = false;

  type QueryWeight = {
    query: string;
    weight: number;
  };

  type MatchResult = {
    text: string;
    score: number;
    offset: [number, number];
    queryWeights: QueryWeight[];
  };

  type DocumentResult = {
    document: string;
    matches: MatchResult[];
  };

  function exportJSON() {
    // Create sanitized versions of all objects
    const sanitizedResults = searchResultSet.results.map(
      ([filename, results]) => [
        filename,
        results.map((result) => ({
          text: result.text,
          distance: result.distance,
          offset: [...result.offset],
          index: result.index,
          filename: result.filename,
          queries: result.queries.map((q) => ({
            query: q.query,
            weight: q.weight,
          })),
          // Omit any properties that might contain PDF references
        })),
      ],
    );

    const exportData = {
      searchTerms: getSearchTerms(),
      results: sanitizedResults,
      sort: searchResultSet.sort,
    };

    // Generate and download the JSON file
    const json = JSON.stringify(exportData, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = generateFileName("json");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  }

  function exportCSV(): void {
    const searchTerms = getSearchTerms();
    const rows: string[][] = [];

    if (searchTerms.length > 0) {
      rows.push(["Search Terms"]);
      rows.push([searchTerms.join(", ")]);
      rows.push([]);
    }

    // Header row
    rows.push([
      "Document",
      "Index",
      "Passage",
      "Score",
      "Offset Start",
      "Offset End",
      "Queries (with weights)",
    ]);

    scoredSearchResultSet.forEach(([filename, results]) => {
      const file = filesByPath[filename];

      results.forEach((result) => {
        const queries = result.queries
          .map((q) => `${q.query} (${q.weight.toFixed(3)})`)
          .join("; ");

        const preferences = result.preferences?.join("; ") ?? "";

        rows.push([
          file?.basename ?? filename,
          result.index.toString(),
          result.text.trim().replace(/[\n\r]+/g, " "),
          result.distance.toFixed(3),
          result.offset?.[0]?.toString() ?? "",
          result.offset?.[1]?.toString() ?? "",
          result.preferences?.length
            ? result.preferences
                .map((p) => `${p.query} (${p.weight})`)
                .join("; ")
            : queries,
        ]);
      });
    });

    const csv = rows
      .map((row) =>
        row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(","),
      )
      .join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = generateFileName("csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  }

  let filterViewed = false;
  let excerptView = false;

  let filenameFilter = "";

  let searchContainer: HTMLDivElement;

  let detailReverse = false;
  let filenameDetailClosed: { [filename: string]: boolean } = {};

  export function scrollToTop() {
    if (searchContainer) searchContainer.scrollTop = 0;
  }

  function filterFilename(filenameFilter: string, filename: string): boolean {
    return filename.toLowerCase().includes(filenameFilter.toLowerCase());
  }

  function getScore(searchResults: SearchResult[]): number {
    let total = 0;
    for (const searchResult of searchResults) {
      total += searchResult.distance;
    }
    return total / searchResults.length;
  }

  $: scoredSearchResultSet = searchResultSet.results
    .map<ScoredSearchResult>(([filename, searchResults]) => [
      filename,
      searchResults,
      getScore(searchResults),
    ])
    .sort((a, b) =>
      searchResultSet.sort === "asc" ? a[2] - b[2] : b[2] - a[2],
    )
    .filter(([filename]) => {
      if (filterViewed && activeFile != null) {
        return filename === activeFile.filename;
      } else {
        return true;
      }
    });
  $: sortedSearchResults = searchResultSet.results
    .map((x) => x[1])
    .flat()
    .sort((a, b) =>
      searchResultSet.sort === "asc"
        ? a.distance - b.distance
        : b.distance - a.distance,
    )
    .filter((searchResult) => {
      if (filterViewed && activeFile != null) {
        return searchResult.filename === activeFile.filename;
      } else {
        return true;
      }
    });

  function handleToggle(e: Event, filename: string) {
    const open = (e.target as HTMLDetailsElement).open;
    filenameDetailClosed[filename] = detailReverse ? open : !open;
  }

  let showExportMenu = false;

  function getSearchTerms(): string[] {
    // Get unique search terms from queries
    return [
      ...new Set(
        searchResultSet.results.flatMap(([_, results]) =>
          results.flatMap((r) => r.queries.map((q) => q.query)),
        ),
      ),
    ];
  }

  function generateFileName(extension: string): string {
    const searchTerms = getSearchTerms();
    const baseFileName =
      searchTerms.length > 0
        ? searchTerms[0]
            .slice(0, 30)
            .replace(/[^a-z0-9]/gi, "_")
            .toLowerCase()
        : `semantra-export-${new Date().toISOString().split("T")[0]}`;

    return `${baseFileName}.${extension}`;
  }
</script>

<div
  class="absolute top-11 z-10 hidden max-sm:block"
  class:hide={sidebarExpanded}
>
  <button
    class="button hamburger-icon"
    title="Toggle sidebar expanded"
    on:click={() => (sidebarExpanded = !sidebarExpanded)}
    >Toggle sidebar expanded</button
  >
</div>
<div
  class="w-1/3 max-lg:w-64 bg-slate-100 max-sm:absolute max-sm:left-0 max-sm:right-8 max-sm:bottom-0 max-sm:top-0 max-sm:w-[calc(100%-8rem)] border-r-4 z-10 border-black flex flex-col items-stretch flex-shrink-0"
  class:hide={!sidebarExpanded}
>
  <div class="flex items-center mb-2 pr-2 max-lg:flex-wrap">
    <div class="mt-2 hidden max-sm:block">
      <button
        class="button hamburger-icon"
        title="Toggle sidebar expanded"
        on:click={() => (sidebarExpanded = !sidebarExpanded)}
        >Toggle sidebar expanded</button
      >
    </div>
    <div class="flex-1 flex items-center relative px-2 mt-2" class:unsearched>
      <input
        class="border border-black bg-white py-1 pl-8 font-mono rounded flex-1 w-40"
        placeholder="Filter files"
        bind:value={filenameFilter}
      />
      <div class="filter-icon">Filter</div>
    </div>
    <div class="mt-2" class:unsearched>
      {#if !excerptView}
        <button
          class="button toggle-detail-icon"
          title="Toggle search results expanded/collapsed"
          on:click={() => {
            if (detailReverse) {
              detailReverse = false;
              filenameDetailClosed = {};
            } else {
              detailReverse = true;
              filenameDetailClosed = {};
            }
          }}
          >{#if detailReverse}Collapse all{:else}Expand all{/if}</button
        >
      {/if}
      <button
        class="button solo-icon"
        class:button-active={filterViewed}
        disabled={activeFile == null}
        on:click={() => (filterViewed = !filterViewed)}
        title={filterViewed ? "Show all files" : "Filter to only viewed file"}
      >
        {#if filterViewed}
          Show all files
        {:else}Filter to only viewed file
        {/if}</button
      >
      <button
        class="button toggle-view-icon"
        title="Toggle search results view"
        on:click={() => (excerptView = !excerptView)}
      >
        {#if excerptView}
          Show file view
        {:else}Show exercept view{/if}</button
      >
      <div class="relative inline-block" id="export-dropdown">
        <button
          class="button save-icon"
          title="Export results"
          on:click|stopPropagation={() => (showExportMenu = !showExportMenu)}
          >Export results</button
        >
        {#if showExportMenu}
          <div
            class="absolute right-0 top-full mt-1 py-2 w-48 bg-white rounded-md shadow-xl z-20 border border-black"
          >
            <button
              class="block px-4 py-2 text-sm w-full text-left hover:bg-gray-100"
              title="Save results to json"
              on:click={() => {
                exportJSON();
                showExportMenu = false;
              }}
            >
              Export as JSON
            </button>
            <button
              class="block px-4 py-2 text-sm w-full text-left hover:bg-gray-100"
              on:click={() => {
                exportCSV();
                showExportMenu = false;
              }}
            >
              Export as CSV
            </button>
          </div>
        {/if}
      </div>
    </div>
  </div>
  <div class="flex-1 relative">
    <div
      class="absolute left-0 top-0 right-0 bottom-0 break-words overflow-y-auto pb-2"
      bind:this={searchContainer}
    >
      {#if excerptView}
        <!-- Excerpt view -->
        <ul class="-mt-2">
          {#each sortedSearchResults as searchResult}
            {@const file = filesByPath[searchResult.filename]}
            {#if file && filterFilename(filenameFilter, file.basename)}
              {#key searchResult}
                <SearchResultComponent
                  on:navigate
                  on:setPreference
                  {file}
                  {searchResult}
                  {preferences}
                  showFilename={true}
                />
              {/key}
            {/if}
          {/each}
        </ul>
      {:else}
        <!-- File view -->
        {#each scoredSearchResultSet as [filename, searchResults, score]}
          {@const file = filesByPath[filename]}
          {#if file && filterFilename(filenameFilter, file.basename)}
            {#key [filename, searchResults, score]}
              <details
                open={detailReverse
                  ? filenameDetailClosed[filename]
                  : !filenameDetailClosed[filename]}
                on:toggle={(e) => handleToggle(e, filename)}
              >
                <summary
                  class="font-mono font-bold cursor-pointer select-none px-2 pt-2 top-0 sticky bg-slate-100"
                >
                  {file ? file.basename : "Unknown file"}
                  <span class="text-xs highlight px-1 rounded"
                    >{score.toFixed(2)}</span
                  >
                </summary>
                <ul class="-mt-2">
                  {#each searchResults as searchResult}
                    {#key searchResult}
                      <SearchResultComponent
                        on:navigate
                        on:setPreference
                        {file}
                        {searchResult}
                        {preferences}
                        showFilename={false}
                      />
                    {/key}
                  {/each}
                </ul>
              </details>
            {/key}
          {/if}
        {/each}
      {/if}
      {#if unsearched}
        <div class="m-2 font-mono">
          Enter a search query above and click the search icon or type “Enter”.
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .highlight {
    background: rgb(255 222 0 / 39%);
  }

  .filter-icon {
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOSIgaGVpZ2h0PSIxNSIgZmlsbD0ibm9uZSI+PHBhdGggc3Ryb2tlPSIjMDAwIiBzdHJva2Utd2lkdGg9IjIiIGQ9Ik04LjYyNSAxMy44MzNoMS4yMk0uMjgyIDEuNTQyaDE3LjkwN002Ljk1NiA5LjczNmg0LjU1OE0zLjY2IDUuNjM5aDExLjE1Ii8+PC9zdmc+");
    background-repeat: no-repeat;
    text-indent: -9999px;
    width: 20px;
    position: absolute;
    left: 16px;
    top: 8px;
  }

  .button {
    @apply border border-black rounded bg-white p-1;
    text-indent: -9999px;
    width: 42px;
    margin: 0 0 0 4px;
    background-position: center;
    background-repeat: no-repeat;
  }

  .button-active {
    @apply bg-gray-200;
  }

  .solo-icon {
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIxMyIgZmlsbD0ibm9uZSI+PHBhdGggc3Ryb2tlPSIjMDAwIiBkPSJNMTAuMTMzIDEuNDhjLTMuNTg1IDAtNC44MzQgMS40NzMtOC45MDggNS4yMDcgNC4wNzQgMy43MzQgNS4zMjMgNS4yMDggOC45MDggNS4yMDggMy41ODUgMCA0LjgzNC0xLjQ3NCA4LjkwOC01LjIwOC00LjA3NC0zLjczNC01LjMyMy01LjIwNy04LjkwOC01LjIwN1oiLz48Y2lyY2xlIGN4PSIxMC4xMzMiIGN5PSI2LjY4NyIgcj0iMi4zMDUiIGZpbGw9IiMwMDAiLz48Y2lyY2xlIGN4PSIxMC4xMzMiIGN5PSI2LjY4NyIgcj0iMy42OTQiIHN0cm9rZT0iIzAwMCIvPjwvc3ZnPg==");
    background-size: 80%;
  }

  .toggle-view-icon {
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIxNiIgZmlsbD0ibm9uZSI+PHBhdGggZmlsbD0iIzAwMCIgZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNLjIgMWgxMHYxSC4zVjFabTEwLjMgOC45aDguNlYxMWgtOC42VjkuOVptLS4yLTdIMi44VjRoNy41VjNabS4yIDguOWg4LjZ2MWgtOC42di0xWm0tLjItN0gyLjhWNmg3LjVWNVptLjIgOC45aDguNnYxaC04LjZ2LTFabS0uMi03SDIuOHYxLjFoNy41di0xWk0xNCA0LjZsLjkgMS0xIC45LTItMi4zLS4zLS41LjUtLjRMMTQgMS42bC44IDEtLjguN0E1IDUgMCAwIDEgMTcgNC42YzEgMSAxLjMgMi4zIDEuMyAzLjhoLTEuM2MwLTEuMy0uMy0yLjItMS0yLjktLjQtLjQtMS0uOC0xLjgtMVptLTggNy4yIDEgMWMtLjktLjItMS41LS41LTEuOS0xLS42LS42LTEtMS42LTEtMi44SDNjMCAxLjQuMyAyLjggMS4zIDMuOEE1IDUgMCAwIDAgNyAxNGwtLjguNy45IDFMOSAxNGwuNC0uNC0uNC0uNS0yLTIuMy0xIC45WiIgY2xpcC1ydWxlPSJldmVub2RkIi8+PC9zdmc+");
    background-size: 70%;
  }

  .save-icon {
    background-image: url("../download.png");
    background-size: 70%;
  }

  .toggle-detail-icon {
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOSIgaGVpZ2h0PSIxNSIgZmlsbD0ibm9uZSI+PHBhdGggc3Ryb2tlPSIjMDAwIiBzdHJva2Utd2lkdGg9IjIiIGQ9Ik01LjIuNHY5LjdNLjQgNS4zSDEwTTkuMiAxMy43aDguOSIvPjwvc3ZnPg==");
    background-size: 60%;
  }

  .hamburger-icon {
    width: 32px;
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOSIgaGVpZ2h0PSIxMSIgZmlsbD0ibm9uZSI+PHBhdGggc3Ryb2tlPSIjMDAwIiBzdHJva2Utd2lkdGg9IjIiIGQ9Ik0uMyAxLjZoMTcuOU0uMyA5LjhoMTcuOU0uMyA1LjdoMTcuOSIvPjwvc3ZnPg==");
    background-size: 80%;
  }

  .hide {
    display: none;
  }

  @media (max-width: 640px) {
    .hide {
      display: block !important;
    }
  }

  .unsearched {
    @apply pointer-events-none opacity-20 select-none;
  }
</style>
