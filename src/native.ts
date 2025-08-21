import * as path from 'path';

const bindings = require(path.join(__dirname, '../build/Release/cudajs.node'));

export default bindings;