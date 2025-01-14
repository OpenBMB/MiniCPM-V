/**
 * Configure and register global directives
 */
import ElTableInfiniteScroll from 'el-table-infinite-scroll';

export function setupGlobDirectives(app) {
    app.use(ElTableInfiniteScroll);
}
