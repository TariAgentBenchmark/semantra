<script lang="ts">
  import { createEventDispatcher, tick } from "svelte";
  import type { Preference } from "../types";
  const dispatch = createEventDispatcher();

  export let preferences: { [key: string]: Preference };
  export let searchDisabled = false;
  export let searching = false;

  $: preferenceValues = Object.values(preferences).filter(
    (preference) => preference.weight !== 0
  );

  function getSearchKey(..._reactiveArgs: any[]) {
    return JSON.stringify({
      query: value,
      preferences: preferenceValues || [],
    });
  }

  let value = "";
  let lastSearchKey = getSearchKey();

  let preferenceContainer: HTMLDivElement;

  export async function scrollToBottomOfPreferences() {
    if (preferenceContainer != null) {
      // Wait for a tick so new preferences are reflected
      await tick();
      preferenceContainer.scrollTop = preferenceContainer.scrollHeight;
    }
  }

  $: searchKey = getSearchKey(value, preferenceValues);
  $: searchOutdated = searchKey !== lastSearchKey;

  function search() {
    if (searchDisabled) {
      return;
    }
    dispatch("search", value);
    lastSearchKey = searchKey;
  }
</script>

<div class="flex flex-1 flex-col">
  <div class="flex items-center relative flex-1">
    <input
      class="bg-white py-2 px-4 font-mono w-full rounded border-black border"
      class:bg-yellow-50={searchOutdated}
      class:border-yellow-600={searchOutdated}
      class:border-dashed={searchOutdated}
      placeholder="Search"
      bind:value
      on:keydown={(e) => {
        if (e.key === "Enter" && !searchDisabled) {
          search();
        }
      }}
      disabled={searchDisabled}
    />
    <button
      class="search-button"
      class:search-button--disabled={searchDisabled}
      class:search-button--busy={searching}
      on:click={search}
      disabled={searchDisabled}
    >
      {#if searching}
        <svg
          class="animate-spin h-4 w-4 mr-2"
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
        Searching...
      {:else}
        Search
      {/if}
    </button>
  </div>

  <div
    class="max-h-24 overflow-y-auto -mb-2 mt-2"
    bind:this={preferenceContainer}
  >
    {#each preferenceValues as preference}
      <button
        class="w-64 max-sm:w-40 truncate monospace rounded px-2 inline-block mr-2 mb-2 cursor"
        class:bg-blue-200={preference.weight > 0}
        class:bg-orange-200={preference.weight < 0}
        title={`${preference.file.basename}: ${preference.searchResult.text}`}
        on:click={() => dispatch("setPreference", { ...preference, weight: 0 })}
      >
        {#if preference.weight > 0}
          <span class="text-blue-600 font-bold mr-1">+</span>
        {:else if preference.weight < 0}
          <span class="text-orange-500 font-bold mr-1">-</span>
        {/if}
        {preference.searchResult.text}
      </button>
    {/each}
  </div>
</div>

<style>
  .search-button {
    margin-left: 0.75rem;
    padding: 0.5rem 1.25rem;
    border-radius: 0.25rem;
    border: 2px solid #000;
    background-color: #1f2937;
    color: #fff;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    transition: opacity 0.2s ease, transform 0.1s ease;
  }

  .search-button:not(:disabled):hover {
    transform: translateY(-1px);
    opacity: 0.9;
  }

  .search-button--disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .search-button--busy svg {
    margin-left: 0;
  }
</style>
