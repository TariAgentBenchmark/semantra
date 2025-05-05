<script lang="ts">
  import type { File } from "../types";

  export let files: File[];
  export let index: number;
  export let disabled: boolean;
  export let onDelete: (filename: string) => void = () => {}; // Default no-op function
</script>

<div class="flex flex-row border-b-4 border-black relative h-10">
  <div
    class="absolute left-0 top-0 right-0 bottom-0 overflow-x-auto thin-scroll"
  >
    <div class="inline-flex flex-nowrap flex-row items-center h-full pl-2">
      {#each files as file, i}
        <div class="flex items-center mr-2">
          <button
            {disabled}
            class:active-tab={i === index}
            class="text-xs rounded-l py-1 px-2 border border-transparent border-r-0"
            on:click={() => (index = i)}
          >
            {file.basename}
          </button>
          <button
            class="text-xs rounded-r py-1 px-1 border border-transparent bg-red-100 hover:bg-red-200 text-red-700"
            class:active-tab-delete={i === index}
            title="Delete this file"
            on:click|stopPropagation={() => onDelete(file.filename)}
          >
            ×
          </button>
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .thin-scroll {
    scrollbar-width: thin;
  }

  .active-tab {
    @apply bg-white border-black;
  }

  .active-tab-delete {
    @apply border-black bg-red-200;
  }
</style>
